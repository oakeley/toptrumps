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

class TopTrumpsDeck:
    """Handles deck loading and preprocessing"""
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.cards = []
        self.attributes = []
        self.attribute_stats = {}
        self._load_deck()

    def _load_deck(self):
        try:
            df = pd.read_csv(self.csv_path)
            if 'Individual' not in df.columns and 'Name' in df.columns:
                df = df.rename(columns={'Name': 'Individual'})
            
            if 'Individual' not in df.columns:
                raise ValueError("Deck must have 'Individual' or 'Name' column")
            
            # First pass: identify numeric columns and analyze distributions
            valid_attributes = []
            attribute_stats = {}
            
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
                    
                    # Calculate distribution statistics
                    stats = {
                        'mean': numeric_series.mean(),
                        'std': numeric_series.std(),
                        'min': numeric_series.min(),
                        'max': numeric_series.max(),
                        'range': numeric_series.max() - numeric_series.min(),
                        'variance': numeric_series.var()
                    }
                    attribute_stats[col] = stats
                    
                    # Fill missing values with minimum
                    min_val = numeric_series[numeric_series.notna()].min()
                    df.loc[numeric_series.isna(), col] = min_val
            
            self.attributes = valid_attributes
            self.attribute_stats = attribute_stats
            
            # Normalize values considering distribution characteristics
            for attr in self.attributes:
                numeric_vals = pd.to_numeric(df[attr], errors='coerce')
                stats = attribute_stats[attr]
                
                # Use z-score normalization if distribution is well-behaved
                if stats['std'] > 0 and stats['range'] > 0:
                    df[attr] = ((numeric_vals - stats['mean']) / stats['std']) * 20 + 50
                else:
                    # Fallback to min-max scaling
                    df[attr] = ((numeric_vals - stats['min']) / max(stats['range'], 1e-6)) * 100
                
                # Clip values to 0-100 range
                df[attr] = df[attr].clip(0, 100)
            
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
    def __init__(self, name: str, hmm_models: Optional[Dict[str, HiddenMarkovModel]] = None):
        super().__init__(name)
        self.hmm_models = hmm_models
        self.current_model = None
        self.current_deck = None
        self.attribute_stats = {}
        self.observation_history = []
        
    def set_deck(self, deck: TopTrumpsDeck):
        """Set current deck and corresponding model"""
        self.current_deck = deck
        if self.hmm_models is not None:
            deck_name = Path(deck.csv_path).stem
            self.current_model = self.hmm_models.get(deck_name)

    def choose_attribute(self, card: dict) -> str:
        if self.current_model is None or self.current_deck is None:
            return max(card['stats'].items(), key=lambda x: x[1])[0]
        
        # Get deck-specific statistics
        attributes = self.current_deck.attributes
        attr_indices = {attr: i for i, attr in enumerate(attributes)}
        
        # Convert observation history to sequence
        if self.observation_history:
            try:
                # Predict next best attribute based on history
                sequence = [attr_indices[attr] for attr, _ in self.observation_history]
                next_state = self.current_model.predict(sequence)[-1]
                predicted_attr = self.current_model.states[next_state].name
                
                # Get predicted attribute's stats
                attr_stats = self.current_deck.attribute_stats[predicted_attr]
                predicted_strength = (card['stats'][predicted_attr] - attr_stats['mean']) / attr_stats['std']
                
                # If predicted attribute looks strong, use it
                if predicted_strength > 0.5:
                    return predicted_attr
                
            except Exception as e:
                logger.debug(f"Prediction error: {e}")
        
        # Fallback to statistical approach
        weighted_scores = {}
        for attr, value in card['stats'].items():
            stats = self.current_deck.attribute_stats[attr]
            
            # Calculate z-score for this value
            z_score = (value - stats['mean']) / stats['std'] if stats['std'] > 0 else 0
            
            # Consider:
            # 1. How exceptional this value is (z-score)
            # 2. How variable this attribute tends to be (std/mean ratio)
            # 3. Historical success rate
            variability = stats['std'] / stats['mean'] if stats['mean'] > 0 else 0
            history_weight = self.attribute_stats.get(attr, {'win_rate': 0.5})['win_rate']
            
            weighted_scores[attr] = (
                z_score * 0.4 +                # How exceptional the value is
                variability * 0.3 +            # How discriminative the attribute is
                history_weight * 0.3           # Historical success
            )
        
        chosen_attr = max(weighted_scores.items(), key=lambda x: x[1])[0]
        return chosen_attr
        
    def update_history(self, attribute: str, won: bool):
        """Update observation history with game results"""
        self.observation_history.append((attribute, won))
        if len(self.observation_history) > 10:  # Keep last 10 moves
            self.observation_history.pop(0)

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
    def __init__(self, deck: TopTrumpsDeck, player1: Player, player2: Player, mode: str = 'train'):
        self.deck = deck
        self.player1 = player1
        self.player2 = player2
        self.game_history = []
        self.round_number = 0
        self.mode = mode

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
            
        # Update AI players' history
        if isinstance(current_player, AIPlayer):
            current_player.update_history(selected_attr, winner == current_player)
        
        round_data = {
            'round': self.round_number,
            'current_player': current_player.name,
            'p1_card': p1_card,
            'p2_card': p2_card,
            'selected_attr': selected_attr,
            'winner': winner.name if winner else 'draw'
        }
        self.game_history.append(round_data)
        
        # Only pause during demo mode (when both players are AI)
        if isinstance(self.player1, AIPlayer) and isinstance(self.player2, AIPlayer) and self.mode == 'demo':
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
            if self.mode != 'train':
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

def create_hmm(deck: TopTrumpsDeck) -> HiddenMarkovModel:
    """Create HMM where states represent the best attribute to choose"""
    states = []
    n_attributes = len(deck.attributes)
    
    # Create a state for each attribute
    for i, attr in enumerate(deck.attributes):
        # Initialize distribution based on attribute statistics
        stats = deck.attribute_stats[attr]
        
        # Calculate initial emission probabilities based on value distribution
        # Higher variance means attribute is more discriminative
        variance_weight = stats['variance'] / (stats['mean'] ** 2) if stats['mean'] > 0 else 0
        
        # Create emission distribution favoring this attribute
        dist = {}
        for j in range(n_attributes):
            if j == i:
                dist[j] = 0.5  # 50% chance of emitting its own attribute
            else:
                dist[j] = 0.5 / (n_attributes - 1)  # Evenly split remaining probability
                
        distribution = DiscreteDistribution(dist)
        states.append(State(distribution, name=attr))

    # Create model
    model = HiddenMarkovModel()
    model.add_states(states)
    
    # Initialize transitions based on attribute correlations
    for i, state1 in enumerate(states):
        attr1 = deck.attributes[i]
        
        # Calculate transition probabilities based on attribute relationships
        total_weight = 0
        weights = []
        
        for j, state2 in enumerate(states):
            attr2 = deck.attributes[j]
            if i == j:
                # Favor staying in same state if attribute has been successful
                weight = 0.4
            else:
                # Base transition probability on relative attribute strengths
                stat1 = deck.attribute_stats[attr1]
                stat2 = deck.attribute_stats[attr2]
                
                # Compare distributions
                mean_diff = abs(stat1['mean'] - stat2['mean'])
                var_ratio = max(stat1['variance'], stat2['variance']) / (min(stat1['variance'], stat2['variance']) + 1e-6)
                
                weight = 1.0 / (1.0 + mean_diff * var_ratio)
            
            weights.append(weight)
            total_weight += weight
        
        # Normalize weights to probabilities
        for j, weight in enumerate(weights):
            prob = weight / total_weight
            model.add_transition(state1, states[j], prob)
            
        # Add start probabilities - favor attributes with higher mean values initially
        start_weight = deck.attribute_stats[attr1]['mean'] / 100.0
        model.add_transition(model.start, state1, start_weight)
    
    model.bake()
    return model

async def train_model(decks: List[str], n_games: int = 1000, epochs: int = 5) -> Dict[str, HiddenMarkovModel]:
    """Train HMM models for each deck type"""
    deck_models = {}
    
    print("\nAnalyzing deck characteristics...")
    # First pass: analyze all decks to understand attribute distributions
    for deck_path in decks:
        deck_name = Path(deck_path).stem
        print(f"\nAnalyzing {deck_name}")
        
        try:
            deck = TopTrumpsDeck(deck_path)
            # Create initial model based on deck statistics
            model = create_hmm(deck)
            deck_models[deck_name] = model
            
            # Output initial attribute analysis
            print("\nAttribute Statistics:")
            for attr in deck.attributes:
                stats = deck.attribute_stats[attr]
                print(f"\n{attr}:")
                print(f"  Mean: {stats['mean']:.2f}")
                print(f"  Std Dev: {stats['std']:.2f}")
                print(f"  Range: {stats['range']:.2f}")
                print(f"  Variance: {stats['variance']:.2f}")
                
        except Exception as e:
            logger.error(f"Error analyzing deck {deck_path}: {e}")
            continue
    
    print("\nStarting Training Phase")
    print("=" * 50)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 30)
        
        # Track stats for each deck separately
        deck_stats = {}
        
        for deck_path in decks:
            deck_name = Path(deck_path).stem
            try:
                deck = TopTrumpsDeck(deck_path)
                model = deck_models[deck_name]
                
                print(f"\nTraining on deck: {deck_name}")
                progress_interval = max(1, n_games // len(decks) // 10)
                
                for game_num in range(n_games // len(decks)):
                    p1 = AIPlayer("Player1", {deck_name: model})
                    p2 = AIPlayer("Player2", None)  # Second player uses simple strategy
                    
                    # Set current deck for p1
                    p1.set_deck(deck)
                    
                    game = TopTrumpsGame(deck, p1, p2, mode='train')
                    winner, history = await game.play_game()
                    
                    if deck_name not in deck_stats:
                        deck_stats[deck_name] = {'wins': 0, 'total': 0}
                    
                    deck_stats[deck_name]['total'] += 1
                    if winner == p1:
                        deck_stats[deck_name]['wins'] += 1
                    
                    # Update model with game history
                    if history:
                        sequence = []
                        for round_data in history:
                            attr_idx = deck.attributes.index(round_data['selected_attr'])
                            sequence.append(attr_idx)
                        
                        if len(sequence) > 1:
                            model.fit([sequence], algorithm='baum-welch', min_iterations=1, max_iterations=5)
                    
                    if game_num % progress_interval == 0:
                        win_rate = deck_stats[deck_name]['wins'] / deck_stats[deck_name]['total'] * 100
                        print(f"\rProgress: {game_num/(n_games//len(decks))*100:.1f}% | Win Rate: {win_rate:.1f}%", end="")
                
                print("\n")  # New line after progress
                
                # Output final model parameters
                print("\nState Analysis:")
                for state in model.states:
                    if hasattr(state, 'name') and state.name in deck.attributes:
                        print(f"\nState: {state.name}")
                        attr_idx = deck.attributes.index(state.name)
                        # Show top transition probabilities
                        transitions = []
                        for next_state in model.states:
                            if hasattr(next_state, 'name') and next_state.name in deck.attributes:
                                prob = model.dense_transition_matrix()[attr_idx][deck.attributes.index(next_state.name)]
                                transitions.append((next_state.name, prob))
                        transitions.sort(key=lambda x: x[1], reverse=True)
                        print("Top transitions:")
                        for attr, prob in transitions[:3]:
                            print(f"  -> {attr}: {prob:.3f}")
                
            except Exception as e:
                logger.error(f"Error training on deck {deck_path}: {e}")
                continue
    
    return deck_models

async def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Top Trumps Game')
    parser.add_argument('mode', choices=['train', 'demo', 'human'])
    parser.add_argument('deck', nargs='?', help='Deck file for demo/human mode')
    parser.add_argument('--output', help='Directory to save models', default='.')
    args = parser.parse_args()

    if args.mode == 'train':
        all_decks = [
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
            'Top_Trumps_The_Art_Game.csv',
            'Top_Trumps_the_Big_Bang_Theory.csv',
            'Top_Trumps_The_Muppet_Show.csv',
            'Top_Trumps_the_Simpsons.csv',
            'Top_Trumps_Transformers_Celebrating_30_Years.csv'
        ]
        
        print("\nInitializing Top Trumps Training")
        print(f"Total decks: {len(all_decks)}")
        
        try:
            models = await train_model(all_decks)
            
            output_path = Path(args.output)
            output_path.mkdir(exist_ok=True)
            
            with open(output_path / 'models.pkl', 'wb') as f:
                pickle.dump(models, f)
            print("\nModels saved successfully")
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return

    else:  # demo or human mode
        if not args.deck:
            print("Error: Deck file required for demo/human mode")
            return
            
        try:
            with open('models.pkl', 'rb') as f:
                models = pickle.load(f)
            
            deck = TopTrumpsDeck(args.deck)
            deck_name = Path(args.deck).stem
            
            if args.mode == 'human':
                print("\nStarting Human vs AI game...")
                ai_player = AIPlayer("AI", models)
                ai_player.set_deck(deck)
                game = TopTrumpsGame(
                    deck,
                    HumanPlayer("Human"),
                    ai_player,
                    mode='human'
                )
            else:  # demo mode
                print("\nStarting AI vs AI demo game...")
                p1 = AIPlayer("AI1", models)
                p2 = AIPlayer("AI2", models)
                p1.set_deck(deck)
                p2.set_deck(deck)
                game = TopTrumpsGame(
                    deck,
                    p1,
                    p2,
                    mode='demo'
                )
            
            winner, history = await game.play_game()
            
            print("\nFinal Statistics:")
            print(f"Total rounds played: {len(history)}")
            print(f"Winner: {winner.name if winner else 'Draw'}")
            
        except FileNotFoundError:
            print("Error: Models file not found. Please train models first.")
        except Exception as e:
            logger.error(f"Error during gameplay: {e}")

if __name__ == "__main__":
    asyncio.run(main())
