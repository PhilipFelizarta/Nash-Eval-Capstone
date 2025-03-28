import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.self_play import self_play_game_stochastic, self_play_game_depth2
from core.pgn_eval import plot_wdl_area_chart

if __name__ == "__main__":
	model_a_path = "models/final_model/final_001.h5"
	model_b_path = "models/final_model/final_048.h5"
	game_path = "figures/self_play_games/example_game.pgn"
	chart_path = "figures/self_play_games/example_game_chart.png"
	

	self_play_game_depth2(model_a_path, model_b_path, game_path)
	plot_wdl_area_chart(game_path, chart_path)