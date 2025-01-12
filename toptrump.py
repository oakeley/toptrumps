import numpy as np
import pandas as pd
from pomegranate import HiddenMarkovModel, State, DiscreteDistribution
import random
import pickle
import asyncio
import logging
import argparse
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from copy import deepcopy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingStats:
    """Track training progress and performance"""
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
            if 'Individual' not in df.columns and 'Name' in df.columns:
                df = df.rename(columns={'Name': 'Individual'})
            
            if 'Individual' not in df.columns:
                raise ValueError("Deck must have 'Individual' or 'Name' column")
            
            # First pass: identify numeric columns
            valid_attributes = []
            for col in df.columns:
                if col == 'Individual':
                    continue
                    
                # Replace 'n/a' with np.nan
                df[col] = df[col].replace('n/a', np.nan)
                
                # Try to convert to numeric
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                
                # Check if column is mostly numeric
                if numeric_series.notna().sum() > len(df) * 0.5:
                    valid_attributes.append(col)
                    min_val = numeric_series[numeric_series.notna()].min()
                    df.loc[numeric_series.isna(), col] = min_val
            
            self.attributes = valid_attributes
            
            # Standardize numeric values to 0-100 range
            for attr in self.attributes:
                numeric_vals = pd.to_numeric(df[attr], errors='coerce')
                min_val = numeric_vals.min()
                max_val = numeric_vals.max()
                if max_val != min_val:
                    df[attr] = ((numeric_vals - min_val) / (max_val - min_val)) * 100
            
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
            if value.lower() == "yes":
                return 100.0
            elif value.lower() == "no":
                return 0.0
            try:
                return float(value.replace(',', ''))
            except ValueError:
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
        self.stats = {'wins': 0, 'losses': 0, 'draws': 0}
    
    def choose_attribute(self, card: dict) -> str:
        raise NotImplementedError
        
    def update_stats(self, won: bool, draw: bool = False):
        if draw:
            self.stats['draws'] += 1
        elif won:
            self.stats['wins'] += 1
        else:
            self.stats['losses'] += 1

class AIPlayer(Player):
    """AI player using HMM for decisions"""
    def __init__(self, name: str, hmm_model: Optional[HiddenMarkovModel] = None):
        super().__init__(name)
        self.hmm_model = hmm_model
        self.attribute_stats = {}

    def choose_attribute(self, card: dict) -> str:
        if self.hmm_model is None or not self.attribute_stats:
            return max(card['stats'].items(), key=lambda x: x[1])[0]
            
        # Weight attributes by success rate and current values
        weighted_scores = {}
        for attr, value in card['stats'].items():
            stats = self.attribute_stats.get(attr, {'win_rate': 0.5})
            weighted_scores[attr] = value * (0.5 + stats['win_rate'])
            
        return max(weighted_scores.items(), key=lambda x: x[1])[0]

    def update_attribute_stats(self, attribute: str, won: bool):
        if attribute not in self.attribute_stats:
            self.attribute_stats[attribute] = {'wins': 0, 'total': 0, 'win_rate': 0.5}
        
        stats = self.attribute_stats[attribute]
        stats['total'] += 1
        if won:
            stats['wins'] += 1
        stats['win_rate'] = stats['wins'] / stats['total']

class HumanPlayer(Player):
    """Human player interface"""
    def choose_attribute(self, card: dict) -> str:
        print("\n" + "="*50)
        print(f"Your card: {card['name']}")
        print("-"*50)
        print("Attributes:")
        for i, (attr, value) in enumerate(card['stats'].items(), 1):
            print(f"{i}. {attr}: {value:.1f}")
        print("="*50)
        
        while True:
            try:
                choice = input("\nChoose attribute number (or 'q' to quit): ")
                if choice.lower() == 'q':
                    raise KeyboardInterrupt
                choice = int(choice) - 1
                if 0 <= choice < len(card['stats']):
                    return list(card['stats'].keys())[choice]
            except ValueError:
                pass
            print("Invalid choice. Try again.")

class TopTrumpsGame:
    """Main game logic"""
    def __init__(self, deck: TopTrumpsDeck, player1: Player, player2: Player):
        self.deck = deck
        self.player1 = player1
        self.player2 = player2
        self.game_history = []
        self.round_number = 0

    def _deal_cards(self):
        """Deal cards with shuffle verification"""
        cards = self.deck.cards.copy()
        random.shuffle(cards)
        
        # Verify shuffle quality
        if len(cards) >= 10:
            original_order = [c['name'] for c in self.deck.cards]
            new_order = [c['name'] for c in cards]
            common_sequence = 0
            for i in range(len(cards)-1):
                if original_order.index(new_order[i]) + 1 == original_order.index(new_order[i+1]):
                    common_sequence += 1
            if common_sequence > len(cards) // 4:
                random.shuffle(cards)
        
        mid = len(cards) // 2
        self.player1.hand = cards[:mid]
        self.player2.hand = cards[mid:]

    def _display_round(self, p1_card: dict, p2_card: dict, selected_attr: str):
        """Display round information"""
        print("\n" + "="*60)
        print(f"Round {self.round_number}".center(60))
        print("-"*60)
        print(f"{self.player1.name}'s card: {p1_card['name']:<30} {self.player2.name}'s card: {p2_card['name']}")
        print(f"Selected attribute: {selected_attr}")
        print(f"Values: {self.player1.name}: {p1_card['stats'][selected_attr]:.1f} vs {self.player2.name}: {p2_card['stats'][selected_attr]:.1f}")
        print("-"*60)

    def play_round(self, current_player: Player, opponent: Player) -> Tuple[Optional[Player], dict]:
        """Play a single round"""
        self.round_number += 1
        
        if not self.player1.hand or not self.player2.hand:
            return None, {}

        p1_card = self.player1.hand[0]
        p2_card = self.player2.hand[0]
        
        selected_attr = current_player.choose_attribute(
            p1_card if current_player == self.player1 else p2_card
        )
        
        self._display_round(p1_card, p2_card, selected_attr)
        
        p1_val = p1_card['stats'][selected_attr]
        p2_val = p2_card['stats'][selected_attr]
        
        if abs(p1_val - p2_val) < 1e-10:
            winner = None
            print("Round Draw!")
            self.player1.update_stats(False, draw=True)
            self.player2.update_stats(False, draw=True)
        elif p1_val > p2_val:
            winner = self.player1
            print(f"{self.player1.name} wins the round!")
            self.player1.update_stats(True)
            self.player2.update_stats(False)
        else:
            winner = self.player2
            print(f"{self.player2.name} wins the round!")
            self.player1.update_stats(False)
            self.player2.update_stats(True)
        
        if isinstance(current_player, AIPlayer):
            current_player.update_attribute_stats(selected_attr, winner == current_player)
        
        round_data = {
            'round': self.round_number,
            'current_player': current_player.name,
            'p1_card': p1_card,
            'p2_card': p2_card,
            'selected_attr': selected_attr,
            'winner': winner.name if winner else 'draw'
        }
        self.game_history.append(round_data)
        
        # In demo mode (AI vs AI), pause after each round
        if isinstance(self.player1, AIPlayer) and isinstance(self.player2, AIPlayer):
            input("\nPress Enter to continue...")
        
        return winner, round_data

    async def play_game(self) -> Tuple[Optional[Player], List[dict]]:
        """Play full game"""
        self._deal_cards()
        current_player = self.player1
        rounds_played = 0
        max_rounds = min(100, len(self.deck.cards) * 2)
        
        print("\nGame Started!")
        print(f"{self.player1.name} vs {self.player2.name}")
        print(f"Initial cards: {len(self.player1.hand)} each")
        
        while self.player1.hand and self.player2.hand and rounds_played < max_rounds:
            opponent = self.player2 if current_player == self.player1 else self.player1
            
            try:
                winner, _ = self.play_round(current_player, opponent)
                
                if len(self.player1.hand) > 0 and len(self.player2.hand) > 0:
                    p1_card = self.player1.hand.pop(0)
                    p2_card = self.player2.hand.pop(0)
                
                    if winner == self.player1:
                        self.player1.hand.extend([p1_card, p2_card])
                        current_player = self.player1
                    elif winner == self.player2:
                        self.player2.hand.extend([p1_card, p2_card])
                        current_player = self.player2
                    else:
                        self.player1.hand.append(p1_card)
                        self.player2.hand.append(p2_card)
                        current_player = opponent
                
            except KeyboardInterrupt:
                print("\nGame terminated by user")
                break
            except Exception as e:
                logger.error(f"Error during round: {e}")
                break
            
            rounds_played += 1
            print(f"Cards remaining - {self.player1.name}: {len(self.player1.hand)}, {self.player2.name}: {len(self.player2.hand)}")
        
        # Determine winner
        if len(self.player1.hand) > len(self.player2.hand):
            game_winner = self.player1
        elif len(self.player2.hand) > len(self.player1.hand):
            game_winner = self.player2
        else:
            p1_total = sum(sum(card['stats'].values()) for card in self.player1.hand)
            p2_total = sum(sum(card['stats'].values()) for card in self.player2.hand)
            game_winner = self.player1 if p1_total >= p2_total else self.player2
            
        print("\nGame Over!")
        print(f"Winner: {game_winner.name}")
        print(f"Final Score - {self.player1.name}: {len(self.player1.hand)} cards, {self.player2.name}: {len(self.player2.hand)} cards")
        
        return game_winner, self.game_history

def create_hmm(n_attributes: int) -> HiddenMarkovModel:
    """Create and initialize HMM with proper smoothing"""
    states = []
    
    # Create 3 states (representing different game situations)
    for i in range(3):
        # Initialize distribution with pseudocounts
        dist = {j: 1.0/n_attributes for j in range(n_attributes)}
        distribution = DiscreteDistribution(dist)
        states.append(State(distribution, name=f"State{i}"))

    # Create model
    model = HiddenMarkovModel()
    model.add_states(states)
    
    # Add transitions with equal probabilities
    for state1 in states:
        for state2 in states:
            model.add_transition(state1, state2, 1.0/3)
        model.add_transition(model.start, state1, 1.0/3)
    
    model.bake()
    return model

async def train_model(training_decks: List[str], validation_decks: List[str], 
                     n_games: int = 1000, epochs: int = 5) -> HiddenMarkovModel:
    """Train HMM model using multiple games"""
    # Get number of attributes from first deck
    first_deck = TopTrumpsDeck(training_decks[0])
    n_attributes = len(first_deck.attributes)
    
    model = create_hmm(n_attributes)
    best_model = None
    best_win_rate = 0
    no_improvement_count = 0
    
    print("\nStarting Training Phase")
    print("=" * 50)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 30)
        
        training_stats = TrainingStats()
        
        for deck_path in training_decks:
            try:
                deck = TopTrumpsDeck(deck_path)
                deck_name = Path(deck_path).stem
                
                print(f"\nTraining on deck: {deck_name}")
                progress_interval = max(1, n_games // 20)
                
                for game_num in range(n_games // len(training_decks)):
                    p1 = AIPlayer("Player1", model)
                    p2 = AIPlayer("Player2", None)  # Second player uses simple strategy
                    game = TopTrumpsGame(deck, p1, p2)
                    
                    winner, history = await game.play_game()
                    
                    if history:
                        training_stats.update(history, winner.name if winner else "draw")
                        
                        # Convert history to training sequence
                        sequence = []
                        for round_data in history:
                            attr_idx = deck.attributes.index(round_data['selected_attr'])
                            sequence.append(attr_idx)
                        
                        if len(sequence) > 1:
                            model.fit([sequence], algorithm='baum-welch', 
                                    min_iterations=1, max_iterations=5)
                    
                    if game_num % progress_interval == 0:
                        print(f"\rProgress: {game_num}/{n_games//len(training_decks)} | {training_stats}", end="")
                        
            except Exception as e:
                logger.error(f"Error processing deck {deck_path}: {e}")
                continue
        
        # Validation phase
        print("\n\nValidating model...")
        val_stats = TrainingStats()
        
        for deck_path in validation_decks:
            try:
                deck = TopTrumpsDeck(deck_path)
                for _ in range(50):  # 50 validation games per deck
                    p1 = AIPlayer("Player1", model)
                    p2 = AIPlayer("Player2", None)
                    game = TopTrumpsGame(deck, p1, p2)
                    winner, history = await game.play_game()
                    if history:
                        val_stats.update(history, winner.name if winner else "draw")
            
            except Exception as e:
                logger.error(f"Error validating with deck {deck_path}: {e}")
                continue
        
        print(f"\nValidation Results: {val_stats}")
        
        # Check for improvement
        if val_stats.win_rate > best_win_rate:
            best_win_rate = val_stats.win_rate
            best_model = deepcopy(model)
            print(f"New best model: {val_stats.win_rate:.1f}% win rate")
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        # Early stopping
        if no_improvement_count >= 2:
            print("\nNo improvement for 2 epochs - stopping training")
            break
            
    print("\nTraining Complete!")
    return best_model or model

async def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Top Trumps Game')
    parser.add_argument('mode', choices=['train', 'demo', 'human'])
    parser.add_argument('deck', nargs='?', help='Deck file for demo/human mode')
    parser.add_argument('--output', help='Directory to save model', default='.')
    args = parser.parse_args()

    if args.mode == 'train':
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
            'Top_Trumps_the_Big_Bang_Theory.csv',
            'Top_Trumps_The_Muppet_Show.csv',
            'Top_Trumps_the_Simpsons.csv',
            'Top_Trumps_Transformers_Celebrating_30_Years.csv'
        ]
        
        validation_decks = [
            'Top_Trumps_Skyscrapers.csv',
            'Top_Trumps_Star_Wars_Rise_of_the_Bounty_Hunters.csv',
            'Top_Trumps_Star_Wars_Starships.csv',
            'Top_Trumps_The_Art_Game.csv'
        ]
        
        print("\nInitializing Top Trumps Training")
        print(f"Training decks: {len(training_decks)}")
        print(f"Validation decks: {len(validation_decks)}")
        
        try:
            model = await train_model(training_decks, validation_decks)
            
            output_path = Path(args.output)
            output_path.mkdir(exist_ok=True)
            
            with open(output_path / 'model.pkl', 'wb') as f:
                pickle.dump(model, f)
            print("\nModel saved successfully")
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return

    else:  # demo or human mode
        if not args.deck:
            print("Error: Deck file required for demo/human mode")
            return
            
        try:
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            
            deck = TopTrumpsDeck(args.deck)
            
            if args.mode == 'human':
                print("\nStarting Human vs AI game...")
                game = TopTrumpsGame(
                    deck,
                    HumanPlayer("Human"),
                    AIPlayer("AI", model)
                )
            else:  # demo mode
                print("\nStarting AI vs AI demo game...")
                game = TopTrumpsGame(
                    deck,
                    AIPlayer("AI1", model),
                    AIPlayer("AI2", model)
                )
            
            winner, history = await game.play_game()
            
            print("\nFinal Statistics:")
            print(f"Total rounds played: {len(history)}")
            print(f"Winner: {winner.name if winner else 'Draw'}")
            
        except FileNotFoundError:
            print("Error: Model file not found. Please train models first.")
        except Exception as e:
            logger.error(f"Error during gameplay: {e}")

if __name__ == "__main__":
    asyncio.run(main())