mport numpy as np
from hmmlearn import hmm
import random
import json

# Load and preprocess dataset
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Preprocess card attributes
def preprocess_data(deck):
    cards = deck['cards']
    attributes = list(cards[0]['stats'].keys())
    return cards, attributes

# Simulate a game
def play_game(player1, player2, deck, hmm_model):
    random.shuffle(deck)
    player1_hand, player2_hand = deck[:len(deck)//2], deck[len(deck)//2:]
    turn = 0  # 0: Player 1, 1: Player 2

    while player1_hand and player2_hand:
        p1_card = player1_hand.pop(0)
        p2_card = player2_hand.pop(0)
        
        # Player selects attribute
        if turn == 0:
            selected_attr = player1.choose_attribute(p1_card)
        else:
            selected_attr = player2.choose_attribute(p2_card)

        # Compare cards
        p1_val, p2_val = p1_card['stats'][selected_attr], p2_card['stats'][selected_attr]
        if p1_val > p2_val:
            player1_hand.extend([p1_card, p2_card])  # Player 1 wins
            turn = 0
        elif p2_val > p1_val:
            player2_hand.extend([p1_card, p2_card])  # Player 2 wins
            turn = 1
        # On draw, both cards are discarded

    return len(player1_hand) > len(player2_hand)  # True if Player 1 wins

class OllamaLLM:
    # Handles interactions with the Ollama phi4 model with improved prompting
    def __init__(self, model_name="phi4"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate(self, prompt: str) -> str:
        # Generate response with retry logic and improved error handling
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

# Define player classes
class LLMPlayer:
    def _init_(self, name, ollama_client):
        self.name = name
        self.ollama_client = ollama_client

    def choose_attribute(self, card):
        # Generate attribute selection prompt
        prompt = f"Choose the best attribute for the card: {card}"
        response = self.ollama_client.generate(prompt)
        return response.strip()

# Define HMM model
def train_hmm(training_games, n_states=3):
    model = hmm.MultinomialHMM(n_components=n_states, n_iter=100)
    obs_sequences = [game_to_obs_sequence(game) for game in training_games]
    lengths = [len(seq) for seq in obs_sequences]
    obs_concat = np.concatenate(obs_sequences)
    model.fit(obs_concat, lengths)
    return model

# Helper to convert game to observation sequence
def game_to_obs_sequence(game):
    # Map game data to a sequence of observations (numerical encoding)
    return np.array([[round['selected_attr'], round['outcome']] for round in game])

# Main optimization loop
def optimize_hmm(training_decks, validation_deck):
    best_model = None
    best_win_rate = 0
    for learning_rate in [0.01, 0.1, 0.2]:
        for n_states in range(2, 5):
            # Train HMM
            hmm_model = train_hmm(training_decks, n_states=n_states)
            
            # Simulate validation games
            wins = 0
            for _ in range(1000):  # Simulate 1000 validation games
                if play_game(LLMPlayer("Player1", ollama_client), LLMPlayer("Player2", ollama_client), validation_deck, hmm_model):
                    wins += 1
            
            win_rate = wins / 1000
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_model = hmm_model

    return best_model

# Load datasets
training_decks = [load_data(f'training_deck_{i}.json') for i in range(5)]
validation_deck = load_data('validation_deck.json')

# Optimize and save the best HMM model
best_hmm = optimize_hmm(training_decks, validation_deck)
with open("best_hmm_model.pkl", "wb") as f:
    pickle.dump(best_hmm, f)
