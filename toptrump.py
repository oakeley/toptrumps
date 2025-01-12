import numpy as np
import pandas as pd
from hmmlearn import hmm
import random
import json
import pickle
import asyncio
import httpx
import logging
import argparse
from typing import List, Dict, Tuple, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Disable httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# Progress tracking
class TrainingStats:
    def __init__(self):
        self.total_games = 0
        self.wins_p1 = 0
        self.total_rounds = 0
        self.avg_rounds_per_game = 0
        self.win_rate = 0
        
    def update(self, game_history: list, winner_name: str):
        self.total_games += 1
        self.wins_p1 += 1 if winner_name == "Player1" else 0
        self.total_rounds += len(game_history)
        self.avg_rounds_per_game = self.total_rounds / self.total_games
        self.win_rate = (self.wins_p1 / self.total_games) * 100
        
    def __str__(self):
        return (f"Win Rate: {self.win_rate:.1f}% | "
                f"Avg Rounds/Game: {self.avg_rounds_per_game:.1f} | "
                f"Total Games: {self.total_games}")

class OllamaLLM:
    """Handles interactions with the Ollama phi4 model"""
    def __init__(self, model_name="phi4"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate(self, prompt: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.api_url,
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                )
                
                if response.status_code != 200:
                    raise RuntimeError(f"Ollama API returned status {response.status_code}")
                    
                result = response.json()["response"].strip()
                if not result:
                    raise ValueError("Empty response from LLM")
                    
                return result
                
        except Exception as e:
            logger.error(f"Error generating response from Ollama: {e}")
            raise

class TopTrumpsDeck:
    """Handles deck loading and preprocessing"""
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.cards = []
        self.attributes = []
        self._load_deck()

    def _load_deck(self):
        try:
            df = pd.read_csv(self.csv_path)
            self.attributes = [col for col in df.columns if col != 'Individual']
            
            # Convert deck to list of dictionaries
            for _, row in df.iterrows():
                card = {
                    'name': row['Individual'],
                    'stats': {attr: self._convert_value(row[attr]) 
                            for attr in self.attributes}
                }
                self.cards.append(card)
                
        except Exception as e:
            logger.error(f"Error loading deck from {self.csv_path}: {e}")
            raise

    def _convert_value(self, value) -> float:
        """Convert string values to numeric, handling special cases"""
        if pd.isna(value):
            return 0.0
        if isinstance(value, str):
            # Handle text values like "Yes"/"No"
            if value.lower() == "yes":
                return 1.0
            elif value.lower() == "no":
                return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

class Player:
    """Base player class"""
    def __init__(self, name: str):
        self.name = name
        self.hand: List[dict] = []

    async def choose_attribute(self, card: dict) -> str:
        raise NotImplementedError

class TrainingPlayer(Player):
    """Fast player for training - uses simple heuristic"""
    def __init__(self, name: str):
        super().__init__(name)
        self.attribute_stats = {}  # Track success rate of attributes

    async def choose_attribute(self, card: dict) -> str:
        if not self.attribute_stats:
            # Initialize stats for each attribute
            for attr in card['stats'].keys():
                self.attribute_stats[attr] = {'wins': 1, 'total': 2}  # Laplace smoothing
        
        # Choose attribute with best win rate
        best_attr = max(self.attribute_stats.items(), 
                       key=lambda x: x[1]['wins'] / x[1]['total'])
        return best_attr[0]

    def update_stats(self, attribute: str, won: bool):
        if attribute in self.attribute_stats:
            self.attribute_stats[attribute]['total'] += 1
            if won:
                self.attribute_stats[attribute]['wins'] += 1

class LLMPlayer(Player):
    """AI player using Ollama LLM"""
    def __init__(self, name: str, ollama_client: OllamaLLM):
        super().__init__(name)
        self.ollama_client = ollama_client

    async def choose_attribute(self, card: dict) -> str:
        prompt = self._generate_attribute_prompt(card)
        response = await self.ollama_client.generate(prompt)
        return self._parse_attribute_response(response, list(card['stats'].keys()))

    def _generate_attribute_prompt(self, card: dict) -> str:
        return f"""You are playing Top Trumps. Here is your card:
Name: {card['name']}
Attributes: {json.dumps(card['stats'], indent=2)}

Choose the attribute you think is most likely to win. Reply with just the attribute name."""

    def _parse_attribute_response(self, response: str, valid_attributes: List[str]) -> str:
        # Clean up response and match to valid attributes
        response = response.strip().lower()
        for attr in valid_attributes:
            if attr.lower() in response:
                return attr
        # Default to first attribute if no match found
        return valid_attributes[0]

class HumanPlayer(Player):
    """Human player interface"""
    async def choose_attribute(self, card: dict) -> str:
        print(f"\nYour card: {card['name']}")
        print("Attributes:")
        for i, (attr, value) in enumerate(card['stats'].items(), 1):
            print(f"{i}. {attr}: {value}")
        
        while True:
            try:
                choice = int(input("\nChoose attribute number: ")) - 1
                if 0 <= choice < len(card['stats']):
                    return list(card['stats'].keys())[choice]
            except ValueError:
                pass
            print("Invalid choice. Try again.")

class TopTrumpsGame:
    """Main game logic"""
    def __init__(self, deck: TopTrumpsDeck, player1: Player, player2: Player, hmm_model: Optional[hmm.MultinomialHMM] = None):
        self.deck = deck
        self.player1 = player1
        self.player2 = player2
        self.hmm_model = hmm_model
        self.game_history = []

    def _deal_cards(self):
        random.shuffle(self.deck.cards)
        mid = len(self.deck.cards) // 2
        self.player1.hand = self.deck.cards[:mid]
        self.player2.hand = self.deck.cards[mid:]

    async def play_round(self, current_player: Player, opponent: Player) -> Tuple[Player, dict]:
        p1_card = self.player1.hand[0]
        p2_card = self.player2.hand[0]
        
        # Get attribute choice
        selected_attr = await current_player.choose_attribute(
            p1_card if current_player == self.player1 else p2_card
        )
        
        # Compare values
        p1_val = p1_card['stats'][selected_attr]
        p2_val = p2_card['stats'][selected_attr]
        
        # Determine winner
        if p1_val > p2_val:
            winner = self.player1
        elif p2_val > p1_val:
            winner = self.player2
        else:
            winner = None
            
        # Record round
        round_data = {
            'p1_card': p1_card,
            'p2_card': p2_card,
            'selected_attr': selected_attr,
            'winner': winner.name if winner else 'draw'
        }
        self.game_history.append(round_data)
        
        return winner, round_data

    async def play_game(self) -> Tuple[Player, List[dict]]:
        self._deal_cards()
        current_player = self.player1
        
        while self.player1.hand and self.player2.hand:
            winner, round_data = await self.play_round(
                current_player,
                self.player2 if current_player == self.player1 else self.player1
            )
            
            # Move cards
            p1_card = self.player1.hand.pop(0)
            p2_card = self.player2.hand.pop(0)
            
            if winner == self.player1:
                self.player1.hand.extend([p1_card, p2_card])
                current_player = self.player1
            elif winner == self.player2:
                self.player2.hand.extend([p1_card, p2_card])
                current_player = self.player2
            # On draw, cards are discarded
            
            # Update HMM if available
            if self.hmm_model:
                self._update_hmm_state(round_data)
        
        game_winner = self.player1 if len(self.player1.hand) > len(self.player2.hand) else self.player2
        return game_winner, self.game_history

    def _update_hmm_state(self, round_data: dict):
        # Convert round data to observation sequence
        obs = self._round_to_observation(round_data)
        if self.hmm_model:
            self.hmm_model.predict(obs.reshape(1, -1))

    def _round_to_observation(self, round_data: dict) -> np.ndarray:
        # Convert round data to numerical observation
        attr_idx = list(self.deck.attributes).index(round_data['selected_attr'])
        outcome = 1 if round_data['winner'] == self.player1.name else 0
        return np.array([attr_idx, outcome])

async def train_model(training_decks: List[str], n_games: int = 1000, epochs: int = 5) -> hmm.MultinomialHMM:
    """Train HMM model using multiple games with fast training players"""
    best_model = None
    best_win_rate = 0
    n_states = 5  # Increased number of states for better modeling
    
    print("\nStarting Training:")
    print("=" * 50)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 50)
        
        # Initialize new model with proper multinomial emission
        n_attributes = len(TopTrumpsDeck(training_decks[0]).attributes)
        model = hmm.MultinomialHMM(
            n_components=n_states,
            n_iter=100,
            implementation="scaling",
            random_state=epoch
        )
        
        all_observations = []
        stats = TrainingStats()
        
        games_per_deck = n_games // len(training_decks)
        total_games = len(training_decks) * games_per_deck
        
        for deck_idx, deck_path in enumerate(training_decks):
            deck = TopTrumpsDeck(deck_path)
            deck_name = Path(deck_path).stem
            p1 = TrainingPlayer("Player1")  # Use fast training player
            p2 = TrainingPlayer("Player2")
            
            print(f"\nTraining on deck: {deck_name}")
            
            for game_num in range(games_per_deck):
                game = TopTrumpsGame(deck, p1, p2)
                winner, history = await game.play_game()
                
                # Update player stats for better attribute selection
                for round_data in history:
                    attr = round_data['selected_attr']
                    if round_data['winner'] == 'Player1':
                        p1.update_stats(attr, True)
                        p2.update_stats(attr, False)
                    elif round_data['winner'] == 'Player2':
                        p1.update_stats(attr, False)
                        p2.update_stats(attr, True)
                
                stats.update(history, winner.name)
                
                observations = np.array([game._round_to_observation(round_data) 
                                      for round_data in history])
                all_observations.append(observations)
                
                # Update progress every 10 games
                if (game_num + 1) % 10 == 0:
                    progress = ((deck_idx * games_per_deck + game_num + 1) / total_games) * 100
                    print(f"\rProgress: {progress:.1f}% | {stats}", end="")
        
        # Fit model with concatenated observations
        if all_observations:
            obs_matrix = np.concatenate(all_observations)
            lengths = [len(obs) for obs in all_observations]
            try:
                model.fit(obs_matrix, lengths)
                
                # Evaluate model
                if stats.win_rate > best_win_rate:
                    best_win_rate = stats.win_rate
                    best_model = model
                    print(f"\n\nNew best model found! Win Rate: {best_win_rate:.1f}%")
                else:
                    print(f"\n\nNo improvement. Current best win rate: {best_win_rate:.1f}%")
            except Exception as e:
                print(f"\nWarning: Model fitting failed: {e}")
                continue
    
    if best_model is None:
        raise RuntimeError("Failed to train any successful models")
    
    print("\nTraining Complete!")
    print(f"Final Best Model Win Rate: {best_win_rate:.1f}%")
    return best_model

async def main():
    parser = argparse.ArgumentParser(description='Top Trumps Game')
    parser.add_argument('mode', choices=['train', 'demo', 'human'])
    parser.add_argument('deck', nargs='?', help='Deck file for demo/human mode')
    args = parser.parse_args()

    if args.mode == 'train':
        # Training mode
        training_decks = [
            'Top_Trumps_Baby_Animals.csv',
            'Top_Trumps_Cats.csv',
            'Top_Trumps_Chicago_JSM2016.csv',
            'Top_Trumps_Dinosaurs.csv',
            'Top_Trumps_Dogs.csv',
            'Top_Trumps_Dr_Who_45_Years_of_Time_Travel.csv',
            'Top_Trumps_Elements.csv',
            'Top_Trumps_Famous_Art_Robberies.csv',
            'Top_Trumps_Harry_Potter_and_the_Deathly_Hallows_Part_2.csv',
            'Top_Trumps_New_York_City.csv',
            'Top_Trumps_Seattle_JSM2015.csv',
            'Top_Trumps_Skyscrapers.csv',
            'Top_Trumps_Star_Wars_Rise_of_the_Bounty_Hunters.csv',
            'Top_Trumps_Star_Wars_Starships.csv',
            'Top_Trumps_The_Art_Game.csv'
        ]
        print("\nInitializing Top Trumps Training")
        print("Using first 15 decks for training")
        model = await train_model(training_decks, n_games=1000, epochs=5)
        
        model_path = 'best_hmm_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"\nBest model saved to {model_path}")

    else:
        # Load model for demo/human mode
        with open('best_hmm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        deck = TopTrumpsDeck(args.deck)
        ollama = OllamaLLM()
        
        if args.mode == 'human':
            game = TopTrumpsGame(
                deck,
                HumanPlayer("Human"),
                LLMPlayer("AI", ollama),
                model
            )
        else:  # demo mode
            game = TopTrumpsGame(
                deck,
                LLMPlayer("AI1", ollama),
                LLMPlayer("AI2", ollama),
                model
            )
        
        winner, history = await game.play_game()
        logger.info(f"Game complete. Winner: {winner.name}")

if __name__ == "__main__":
    asyncio.run(main())
