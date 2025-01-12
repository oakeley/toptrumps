import numpy as np
import pandas as pd
from hmmlearn import hmm
from hmmlearn.hmm import CategoricalHMM
import random
import json
import pickle
import asyncio
import logging
from copy import deepcopy
import argparse
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential


# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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
            
            # Handle both 'Individual' and 'Name' columns
            name_col = 'Individual'
            if 'Individual' not in df.columns and 'Name' in df.columns:
                name_col = 'Name'
            
            if name_col not in df.columns:
                raise ValueError(f"CSV must have either 'Individual' or 'Name' column")
            
            self.attributes = [col for col in df.columns if col != name_col]
            
            # Check for numeric convertibility and log warnings
            for attr in self.attributes:
                non_numeric = pd.to_numeric(df[attr], errors='coerce').isna()
                if non_numeric.any():
                    logger.warning(f"Column {attr} contains non-numeric values")
                    logger.debug(f"Non-numeric values in {attr}: {df[attr][non_numeric].unique()}")
            
            # Convert deck to list of dictionaries
            for _, row in df.iterrows():
                card = {
                    'name': str(row[name_col]),  # Ensure name is string
                    'stats': {attr: self._convert_value(row[attr]) 
                            for attr in self.attributes}
                }
                self.cards.append(card)
                
            if not self.cards:
                raise ValueError("No cards were loaded from the deck")
                
            logger.info(f"Successfully loaded {len(self.cards)} cards with {len(self.attributes)} attributes")
                
        except pd.errors.EmptyDataError:
            logger.error(f"CSV file {self.csv_path} is empty")
            raise
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing CSV file {self.csv_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading deck from {self.csv_path}: {e}")
            raise

    def _convert_value(self, value) -> float:
        """Convert string values to numeric, handling special cases"""
        if pd.isna(value):
            return 0.0
        if isinstance(value, str):
            # Handle various text cases
            value = value.lower().strip()
            if value in ('yes', 'true', 'y'):
                return 1.0
            elif value in ('no', 'false', 'n'):
                return 0.0
            elif value.endswith('%'):
                try:
                    return float(value.rstrip('%')) / 100
                except ValueError:
                    pass
            # Remove any commas and try to convert
            value = value.replace(',', '')
            try:
                return float(value)
            except ValueError:
                # If conversion fails, log at debug level and return 0
                logger.debug(f"Could not convert value '{value}' to float")
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

    async def choose_attribute(self, card: dict) -> str:
        raise NotImplementedError
        
    def update_stats(self, won: bool, draw: bool = False):
        if draw:
            self.stats['draws'] += 1
        elif won:
            self.stats['wins'] += 1
        else:
            self.stats['losses'] += 1

class TrainingPlayer(Player):
    """Player using HMM model for decisions"""
    def __init__(self, name: str, hmm_model: Optional[CategoricalHMM] = None):
        super().__init__(name)
        self.hmm_model = hmm_model
        self.state_history = []
        self.attribute_stats = {}  # Track success rate of each attribute

    def update_attribute_stats(self, attribute: str, won: bool):
        """Track attribute performance"""
        if attribute not in self.attribute_stats:
            self.attribute_stats[attribute] = {'wins': 0, 'total': 0, 'win_rate': 0.0}
        
        stats = self.attribute_stats[attribute]
        stats['total'] += 1
        if won:
            stats['wins'] += 1
        stats['win_rate'] = stats['wins'] / stats['total']

    async def choose_attribute(self, card: dict) -> str:
        if self.hmm_model is None:
            # Fallback to highest value strategy
            return max(card['stats'].items(), key=lambda x: x[1])[0]
            
        try:
            # Use HMM to predict best attribute
            if self.state_history:
                state_array = np.array(self.state_history).reshape(-1, 1)
                current_state = self.hmm_model.predict(state_array)[-1]
            else:
                current_state = 0
                
            # Get probabilities from the model
            if hasattr(self.hmm_model, 'transmat_'):
                # Use transition probabilities if available
                probs = self.hmm_model.transmat_[current_state]
            else:
                # Otherwise use uniform distribution
                n_attributes = len(card['stats'])
                probs = np.ones(n_attributes) / n_attributes
            
            # Ensure index is within bounds
            attributes = list(card['stats'].keys())
            n_attributes = len(attributes)
            
            # Weight probabilities by attribute values
            weighted_probs = np.array([card['stats'][attr] for attr in attributes])
            weighted_probs = weighted_probs * probs[:n_attributes]
            attr_idx = np.argmax(weighted_probs)
            
            # Update state history
            self.state_history.append([attr_idx])
            
            return attributes[attr_idx]
            
        except Exception as e:
            logger.error(f"Error in attribute selection: {e}")
            # Fallback to highest value if there's an error
            return max(card['stats'].items(), key=lambda x: x[1])[0]

async def train_models(training_decks: List[str], validation_decks: List[str], 
                      n_games: int = 50, epochs: int = 5) -> Tuple[CategoricalHMM, CategoricalHMM]:
    """Train two HMM models adversarially"""
    
    # Get maximum number of attributes across all decks
    max_attributes = 0
    for deck_path in training_decks + validation_decks:
        try:
            deck = TopTrumpsDeck(deck_path)
            max_attributes = max(max_attributes, len(deck.attributes))
        except Exception as e:
            logger.warning(f"Could not process deck {deck_path}: {e}")
            continue
    
    # Initialize models with correct number of categories
    model1 = CategoricalHMM(
        n_components=5,  # number of hidden states
        n_iter=100,
        random_state=42
    )
    model2 = CategoricalHMM(
        n_components=5,
        n_iter=100,
        random_state=43
    )
    
    # Initialize parameters for both models
    n_states = 5
    startprob = np.ones(n_states) / n_states
    transmat = np.ones((n_states, n_states)) / n_states
    emissionprob = np.ones((n_states, max_attributes)) / max_attributes
    
    # Set initial parameters for both models
    model1.startprob_ = startprob
    model1.transmat_ = transmat
    model1.emissionprob_ = emissionprob
    
    model2.startprob_ = startprob.copy()
    model2.transmat_ = transmat.copy()
    model2.emissionprob_ = emissionprob.copy()
    
    best_model1, best_model2 = None, None
    best_win_rate1, best_win_rate2 = 0, 0
    
    print(f"\nInitialized models with {max_attributes} possible attributes")
    print("\nPhase 1: Initial Adversarial Training")
    print("=" * 50)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 30)
        
        # Training statistics
        train_stats = {'p1_wins': 0, 'p2_wins': 0, 'total_games': 0}
        
        # Training phase
        for deck_path in training_decks:
            try:
                deck = TopTrumpsDeck(deck_path)
                deck_name = Path(deck_path).stem
                
                for game_num in range(n_games):
                    p1 = TrainingPlayer("Player1", model1)
                    p2 = TrainingPlayer("Player2", model2)
                    game = TopTrumpsGame(deck, p1, p2)
                    winner, history = await game.play_game()
                    
                    train_stats['total_games'] += 1
                    if winner.name == "Player1":
                        train_stats['p1_wins'] += 1
                    elif winner.name == "Player2":
                        train_stats['p2_wins'] += 1
                    
                    # Update models
                    if history:
                        try:
                            # Convert history to observation sequences
                            observations = []
                            for round_data in history:
                                attr_idx = list(deck.attributes).index(round_data['selected_attr'])
                                observations.append([attr_idx])
                            
                            if observations:
                                obs_array = np.array(observations)
                                lengths = [len(observations)]
                                
                                # Fit models with error handling
                                try:
                                    model1.fit(obs_array, lengths)
                                except Exception as e:
                                    logger.warning(f"Model 1 fitting failed: {e}")
                                    
                                try:
                                    model2.fit(obs_array, lengths)
                                except Exception as e:
                                    logger.warning(f"Model 2 fitting failed: {e}")
                                    
                        except Exception as e:
                            logger.warning(f"Error processing game history: {e}")
                            continue
                    
                    # Print progress
                    if game_num % 10 == 0:
                        p1_rate = (train_stats['p1_wins'] / train_stats['total_games']) * 100
                        p2_rate = (train_stats['p2_wins'] / train_stats['total_games']) * 100
                        print(f"\rTraining: {deck_name} | Game {game_num}/{n_games} | "
                              f"P1 Rate: {p1_rate:.1f}% | P2 Rate: {p2_rate:.1f}%", end="")
                              
            except Exception as e:
                logger.error(f"Error processing deck {deck_path}: {e}")
                continue
#############
class TrainingPlayer(Player):
    """Player using HMM model for decisions"""
    def __init__(self, name: str, hmm_model: Optional[hmm.CategoricalHMM] = None):
        super().__init__(name)
        self.hmm_model = hmm_model
        self.state_history = []

    async def choose_attribute(self, card: dict) -> str:
        if self.hmm_model is None:
            # Fallback to highest value strategy
            return max(card['stats'].items(), key=lambda x: x[1])[0]
            
        try:
            # Use HMM to predict best attribute
            if self.state_history:
                state_array = np.array(self.state_history).reshape(-1, 1)
                current_state = self.hmm_model.predict(state_array)[-1]
            else:
                current_state = 0
                
            # Use emission probabilities to choose attribute
            emission_probs = self.hmm_model.emissionprob_[current_state]
            
            # Ensure index is within bounds
            attributes = list(card['stats'].keys())
            n_attributes = len(attributes)
            attr_idx = np.argmax(emission_probs[:n_attributes])
            
            # Update state history
            self.state_history.append([attr_idx])
            
            return attributes[attr_idx]
            
        except Exception as e:
            logger.error(f"Error in attribute selection: {e}")
            # Fallback to highest value if there's an error
            return max(card['stats'].items(), key=lambda x: x[1])[0]

async def train_models(training_decks: List[str], validation_decks: List[str], 
                      n_games: int = 50, epochs: int = 5) -> Tuple[hmm.CategoricalHMM, hmm.CategoricalHMM]:
    """Train two HMM models adversarially"""
    
    # Get maximum number of attributes across all decks
    max_attributes = 0
    for deck_path in training_decks + validation_decks:
        try:
            deck = TopTrumpsDeck(deck_path)
            max_attributes = max(max_attributes, len(deck.attributes))
        except Exception as e:
            logger.warning(f"Could not process deck {deck_path}: {e}")
            continue
    
    # Initialize models with correct number of observation symbols
    model1 = hmm.CategoricalHMM(
        n_components=5,
        n_iter=100,
        init_params="ste",
        random_state=42,
        n_trials=1  # Each observation is a single choice
    )
    model2 = hmm.CategoricalHMM(
        n_components=5,
        n_iter=100,
        init_params="ste",
        random_state=43,
        n_trials=1  # Each observation is a single choice
    )
    
    # Initialize emission probabilities manually to handle all possible attributes
    model1.emissionprob_ = np.ones((5, max_attributes)) / max_attributes
    model2.emissionprob_ = np.ones((5, max_attributes)) / max_attributes
    
    best_model1, best_model2 = None, None
    best_win_rate1, best_win_rate2 = 0, 0
    
    print(f"\nInitialized models with {max_attributes} possible attributes")
    print("\nPhase 1: Initial Adversarial Training")
    print("=" * 50)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 30)
        
        # Training statistics
        train_stats = {'p1_wins': 0, 'p2_wins': 0, 'total_games': 0}
        
        # Training phase
        for deck_path in training_decks:
            try:
                deck = TopTrumpsDeck(deck_path)
                deck_name = Path(deck_path).stem
                
                for game_num in range(n_games):
                    p1 = TrainingPlayer("Player1", model1)
                    p2 = TrainingPlayer("Player2", model2)
                    game = TopTrumpsGame(deck, p1, p2)
                    winner, history = await game.play_game()
                    
                    train_stats['total_games'] += 1
                    if winner.name == "Player1":
                        train_stats['p1_wins'] += 1
                    elif winner.name == "Player2":
                        train_stats['p2_wins'] += 1
                    
                    # Update models
                    if history:
                        try:
                            # Convert history to observation sequences
                            observations = []
                            for round_data in history:
                                attr_idx = list(deck.attributes).index(round_data['selected_attr'])
                                observations.append([attr_idx])
                            
                            if observations:
                                obs_array = np.array(observations)
                                lengths = [len(observations)]
                                
                                # Fit models with error handling
                                try:
                                    model1.fit(obs_array, lengths)
                                except Exception as e:
                                    logger.warning(f"Model 1 fitting failed: {e}")
                                    
                                try:
                                    model2.fit(obs_array, lengths)
                                except Exception as e:
                                    logger.warning(f"Model 2 fitting failed: {e}")
                                    
                        except Exception as e:
                            logger.warning(f"Error processing game history: {e}")
                            continue
                    
                    # Print progress
                    if game_num % 10 == 0:
                        p1_rate = (train_stats['p1_wins'] / train_stats['total_games']) * 100
                        p2_rate = (train_stats['p2_wins'] / train_stats['total_games']) * 100
                        print(f"\rTraining: {deck_name} | Game {game_num}/{n_games} | "
                              f"P1 Rate: {p1_rate:.1f}% | P2 Rate: {p2_rate:.1f}%", end="")
                              
            except Exception as e:
                logger.error(f"Error processing deck {deck_path}: {e}")
                continue

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

class LLMPlayer(Player):
    """AI player using Ollama LLM"""
    def __init__(self, name: str):
        super().__init__(name)
        self.ollama_client = OllamaLLM()

    async def choose_attribute(self, card: dict) -> str:
        prompt = f"""You are playing Top Trumps. Here is your card:
Name: {card['name']}
Attributes: {json.dumps(card['stats'], indent=2)}

Choose the attribute you think is most likely to win. Reply with just the attribute name."""
        
        response = await self.ollama_client.generate(prompt)
        
        # Match response to valid attributes
        response = response.strip().lower()
        for attr in card['stats'].keys():
            if attr.lower() in response:
                return attr
        return list(card['stats'].keys())[0]  # Default to first attribute if no match

class TopTrumpsGame:
    """Main game logic with improved mechanics and logging"""
    def __init__(self, deck: TopTrumpsDeck, player1: Player, player2: Player):
        self.deck = deck
        self.player1 = player1
        self.player2 = player2
        self.game_history = []
        self.round_number = 0

    def _deal_cards(self):
        """Deal cards with optional shuffle verification"""
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
                logger.warning("Possible poor shuffle quality detected")
                random.shuffle(cards)  # Reshuffle if needed
        
        mid = len(cards) // 2
        self.player1.hand = cards[:mid]
        self.player2.hand = cards[mid:]

    async def play_round(self, current_player: Player, opponent: Player) -> Tuple[Optional[Player], dict]:
        """Play a single round with improved mechanics"""
        self.round_number += 1
        
        if not self.player1.hand or not self.player2.hand:
            return None, {}

        p1_card = self.player1.hand[0]
        p2_card = self.player2.hand[0]
        
        # Get attribute choice
        selected_attr = await current_player.choose_attribute(
            p1_card if current_player == self.player1 else p2_card
        )
        
        # Validate attribute exists
        if selected_attr not in p1_card['stats'] or selected_attr not in p2_card['stats']:
            logger.error(f"Invalid attribute selected: {selected_attr}")
            selected_attr = list(p1_card['stats'].keys())[0]
        
        # Compare values
        p1_val = p1_card['stats'][selected_attr]
        p2_val = p2_card['stats'][selected_attr]
        
        # Determine winner with tie handling
        if abs(p1_val - p2_val) < 1e-10:  # Handle floating point comparison
            winner = None
            self.player1.update_stats(False, draw=True)
            self.player2.update_stats(False, draw=True)
        elif p1_val > p2_val:
            winner = self.player1
            self.player1.update_stats(True)
            self.player2.update_stats(False)
        else:
            winner = self.player2
            self.player1.update_stats(False)
            self.player2.update_stats(True)
        
        # Update attribute statistics for training players
        if isinstance(current_player, TrainingPlayer):
            current_player.update_attribute_stats(selected_attr, winner == current_player)
        
        # Record round
        round_data = {
            'round': self.round_number,
            'current_player': current_player.name,
            'p1_card': p1_card,
            'p2_card': p2_card,
            'selected_attr': selected_attr,
            'p1_value': p1_val,
            'p2_value': p2_val,
            'winner': winner.name if winner else 'draw'
        }
        self.game_history.append(round_data)
        
        return winner, round_data

    async def play_game(self) -> Tuple[Optional[Player], List[dict]]:
        """Play full game with improved handling"""
        self._deal_cards()
        current_player = self.player1
        rounds_played = 0
        max_rounds = min(100, len(self.deck.cards) * 2)  # Dynamic round limit
        
        while self.player1.hand and self.player2.hand and rounds_played < max_rounds:
            opponent = self.player2 if current_player == self.player1 else self.player1
            
            try:
                winner, round_data = await self.play_round(current_player, opponent)
                
                # Move cards with validation
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
                        # On draw, each player keeps their card
                        self.player1.hand.append(p1_card)
                        self.player2.hand.append(p2_card)
                        # On draw, alternate current player
                        current_player = opponent
                else:
                    break
                
            except Exception as e:
                logger.error(f"Error during round: {e}")
                break
            
            rounds_played += 1
            
            # Periodic progress update
            if rounds_played % 10 == 0:
                logger.debug(f"Round {rounds_played}: P1 cards={len(self.player1.hand)}, " 
                           f"P2 cards={len(self.player2.hand)}")
        
        # Determine game winner
        if len(self.player1.hand) > len(self.player2.hand):
            game_winner = self.player1
        elif len(self.player2.hand) > len(self.player1.hand):
            game_winner = self.player2
        else:
            # In case of exact tie, compare total card values
            p1_total = sum(sum(card['stats'].values()) for card in self.player1.hand)
            p2_total = sum(sum(card['stats'].values()) for card in self.player2.hand)
            game_winner = self.player1 if p1_total >= p2_total else self.player2
            
        return game_winner, self.game_history

def initialize_hmm(n_components: int, n_features: int, random_state: int) -> CategoricalHMM:
    """Initialize a CategoricalHMM with proper parameters"""
    model = CategoricalHMM(
        n_components=n_components,
        n_iter=100,
        random_state=random_state
    )
    
    # Initialize probabilities
    model.startprob_ = np.ones(n_components) / n_components
    model.transmat_ = np.ones((n_components, n_components)) / n_components
    
    # For CategoricalHMM, we need to set n_features
    model.n_features = n_features
    
    return model

async def train_models(training_decks: List[str], validation_decks: List[str], 
                      n_games: int = 50, epochs: int = 5) -> Tuple[CategoricalHMM, CategoricalHMM]:
    """Train two HMM models adversarially"""
    
    # Get maximum number of attributes across all decks
    max_attributes = 0
    for deck_path in training_decks + validation_decks:
        try:
            deck = TopTrumpsDeck(deck_path)
            max_attributes = max(max_attributes, len(deck.attributes))
        except Exception as e:
            logger.warning(f"Could not process deck {deck_path}: {e}")
            continue
    
    # Initialize models
    model1 = initialize_hmm(n_components=5, n_features=max_attributes, random_state=42)
    model2 = initialize_hmm(n_components=5, n_features=max_attributes, random_state=43)
    
    best_model1, best_model2 = None, None
    best_win_rate1, best_win_rate2 = 0, 0
    
    print(f"\nInitialized models with {max_attributes} possible attributes")
    print("\nPhase 1: Initial Adversarial Training")
    print("=" * 50)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 30)
        
        # Training statistics with more metrics
        train_stats = {
            'p1_wins': 0, 'p2_wins': 0, 'draws': 0,
            'total_games': 0, 'rounds_played': 0
        }
        
        # Training phase with improved monitoring
        for deck_path in training_decks:
            try:
                deck = TopTrumpsDeck(deck_path)
                deck_name = Path(deck_path).stem
                
                for game_num in range(n_games):
                    p1 = TrainingPlayer("Player1", model1)
                    p2 = TrainingPlayer("Player2", model2)
                    game = TopTrumpsGame(deck, p1, p2)
                    
                    winner, history = await game.play_game()
                    
                    train_stats['total_games'] += 1
                    train_stats['rounds_played'] += len(history)
                    
                    if winner.name == "Player1":
                        train_stats['p1_wins'] += 1
                    elif winner.name == "Player2":
                        train_stats['p2_wins'] += 1
                    else:
                        train_stats['draws'] += 1
                    
                    # Update models with sequence data
                    if history:
                        observations = np.array([[
                            list(game.deck.attributes).index(round_data['selected_attr']),
                            1 if round_data['winner'] == 'Player1' else 0
                        ] for round_data in history])
                        
                        if len(observations) > 1:  # Need at least 2 observations
                            try:
                                model1.fit(observations)
                                model2.fit(observations)
                            except Exception as e:
                                logger.warning(f"Model fitting failed: {e}")
                    
                    # Print detailed progress
                    if game_num % 5 == 0:
                        total = train_stats['total_games']
                        p1_rate = (train_stats['p1_wins'] / total) * 100
                        p2_rate = (train_stats['p2_wins'] / total) * 100
                        avg_rounds = train_stats['rounds_played'] / total
                        
                        print(f"\rTraining: {deck_name} | Game {game_num}/{n_games} | "
                              f"P1 Rate: {p1_rate:.1f}% | P2 Rate: {p2_rate:.1f}% | "
                              f"Avg Rounds: {avg_rounds:.1f}", end="")
                
            except Exception as e:
                logger.error(f"Error processing deck {deck_path}: {e}")
                continue
        
        # Validation phase with improved metrics
        print("\nValidating models...")
        val_stats = {'p1_wins': 0, 'p2_wins': 0, 'draws': 0, 'total_games': 0}
        
        for deck_path in validation_decks:
            try:
                deck = TopTrumpsDeck(deck_path)
                for _ in range(20):  # 20 validation games per deck
                    p1 = TrainingPlayer("Player1", model1)
                    p2 = TrainingPlayer("Player2", model2)
                    game = TopTrumpsGame(deck, p1, p2)
                    winner, _ = await game.play_game()
                    
                    val_stats['total_games'] += 1
                    if winner.name == "Player1":
                        val_stats['p1_wins'] += 1
                    elif winner.name == "Player2":
                        val_stats['p2_wins'] += 1
                    else:
                        val_stats['draws'] += 1
            
            except Exception as e:
                logger.error(f"Error validating with deck {deck_path}: {e}")
                continue
        
        # Calculate validation metrics
        total_val = val_stats['total_games']
        if total_val > 0:
            val_rate1 = (val_stats['p1_wins'] / total_val) * 100
            val_rate2 = (val_stats['p2_wins'] / total_val) * 100
            draw_rate = (val_stats['draws'] / total_val) * 100
            
            print(f"\nValidation Results:")
            print(f"Model 1: {val_rate1:.1f}% | Model 2: {val_rate2:.1f}% | Draws: {draw_rate:.1f}%")
            
            # Save best models
            if val_rate1 > best_win_rate1:
                best_win_rate1 = val_rate1
                best_model1 = deepcopy(model1)
                print(f"New best for Model 1: {val_rate1:.1f}%")
                
            if val_rate2 > best_win_rate2:
                best_win_rate2 = val_rate2
                best_model2 = deepcopy(model2)
                print(f"New best for Model 2: {val_rate2:.1f}%")
            
            # Print model parameters
            print("\nModel Parameters:")
            print(f"Model 1 - States: {model1.n_components}, "
                  f"Transition matrix shape: {model1.transmat_.shape}")
            print(f"Model 2 - States: {model2.n_components}, "
                  f"Transition matrix shape: {model2.transmat_.shape}")
    
    # Phase 2: Optimize better model
    print("\nPhase 2: Optimizing Better Model")
    print("=" * 50)
    
    # Determine which model to optimize
    if best_win_rate1 >= best_win_rate2:
        optimize_model = deepcopy(best_model1)
        fixed_model = best_model2
        optimize_player = 1
    else:
        optimize_model = deepcopy(best_model2)
        fixed_model = best_model1
        optimize_player = 2
        
    print(f"Optimizing Player {optimize_player} model")
    
    # Additional training epochs for the better model
    for epoch in range(3):
        print(f"\nOptimization Epoch {epoch + 1}/3")
        
        opt_stats = {'wins': 0, 'total': 0, 'rounds': 0}
        
        for deck_path in training_decks[:5]:  # Use subset for optimization
            try:
                deck = TopTrumpsDeck(deck_path)
                
                for game_num in range(n_games):
                    if optimize_player == 1:
                        p1 = TrainingPlayer("Player1", optimize_model)
                        p2 = TrainingPlayer("Player2", fixed_model)
                    else:
                        p1 = TrainingPlayer("Player1", fixed_model)
                        p2 = TrainingPlayer("Player2", optimize_model)
                        
                    game = TopTrumpsGame(deck, p1, p2)
                    winner, history = await game.play_game()
                    
                    opt_stats['total'] += 1
                    opt_stats['rounds'] += len(history)
                    
                    if ((winner.name == "Player1" and optimize_player == 1) or
                        (winner.name == "Player2" and optimize_player == 2)):
                        opt_stats['wins'] += 1
                    
                    # Update optimizing model
                    if history:
                        observations = np.array([[
                            list(game.deck.attributes).index(round_data['selected_attr']),
                            1 if round_data['winner'] == f'Player{optimize_player}' else 0
                        ] for round_data in history])
                        
                        if len(observations) > 1:
                            try:
                                optimize_model.fit(observations)
                            except Exception as e:
                                logger.warning(f"Optimization fitting failed: {e}")
                    
                    if game_num % 10 == 0:
                        win_rate = (opt_stats['wins'] / opt_stats['total']) * 100
                        avg_rounds = opt_stats['rounds'] / opt_stats['total']
                        print(f"\rOptimization progress: {win_rate:.1f}% win rate | "
                              f"Avg rounds: {avg_rounds:.1f}", end="")
                        
            except Exception as e:
                logger.error(f"Error during optimization with deck {deck_path}: {e}")
                continue
    
    print("\n\nTraining Complete!")
    
    # Return models in correct order
    if optimize_player == 1:
        return optimize_model, fixed_model
    else:
        return fixed_model, optimize_model

async def main():
    """Main execution with improved error handling and logging"""
    parser = argparse.ArgumentParser(description='Top Trumps Game')
    parser.add_argument('mode', choices=['train', 'demo', 'human'])
    parser.add_argument('deck', nargs='?', help='Deck file for demo/human mode')
    parser.add_argument('--output', help='Directory to save models', default='.')
    args = parser.parse_args()

    if args.mode == 'train':
        # Define deck lists for training and validation
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
            'Top_Trumps_Seattle_JSM2015.csv'
        ]
        
        validation_decks = [
            'Top_Trumps_Skyscrapers.csv',
            'Top_Trumps_Star_Wars_Rise_of_the_Bounty_Hunters.csv',
            'Top_Trumps_Star_Wars_Starships.csv',
            'Top_Trumps_The_Art_Game.csv'
        ]
        
        print("\nInitializing Top Trumps Training")
        print(f"Using {len(training_decks)} decks for training and {len(validation_decks)} for validation")
        
        try:
            # Train the models
            model1, model2 = await train_models(training_decks, validation_decks)
            
            # Save models with error handling
            output_path = Path(args.output)
            output_path.mkdir(exist_ok=True)
            
            try:
                with open(output_path / 'model1.pkl', 'wb') as f:
                    pickle.dump(model1, f)
                with open(output_path / 'model2.pkl', 'wb') as f:
                    pickle.dump(model2, f)
                print("\nModels saved successfully")
                
            except Exception as e:
                logger.error(f"Error saving models: {e}")
                return
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return

    else:  # demo or human mode
        if not args.deck:
            print("Error: Deck file required for demo/human mode")
            return
            
        try:
            # Load models
            try:
                with open('model1.pkl', 'rb') as f:
                    model1 = pickle.load(f)
                with open('model2.pkl', 'rb') as f:
                    model2 = pickle.load(f)
            except FileNotFoundError:
                print("Error: Models not found. Please train models first.")
                return
            except Exception as e:
                logger.error(f"Error loading models: {e}")
                return
            
            # Initialize game
            try:
                deck = TopTrumpsDeck(args.deck)
                
                if args.mode == 'human':
                    print("\nStarting Human vs AI game...")
                    print("You will play against an AI opponent.")
                    game = TopTrumpsGame(
                        deck,
                        HumanPlayer("Human"),
                        LLMPlayer("AI")
                    )
                else:  # demo mode
                    print("\nStarting AI vs AI demo game...")
                    game = TopTrumpsGame(
                        deck,
                        LLMPlayer("AI1"),
                        LLMPlayer("AI2")
                    )
                
                winner, history = await game.play_game()
                
                # Print game summary
                print(f"\nGame complete!")
                print(f"Winner: {winner.name}")
                print(f"Final hand sizes: {winner.name}: {len(winner.hand)} cards")
                print(f"Total rounds played: {len(history)}")
                
            except Exception as e:
                logger.error(f"Error during gameplay: {e}")
                return
                
        except Exception as e:
            logger.error(f"Error initializing game: {e}")
            return

if __name__ == "__main__":
    asyncio.run(main())
