def get_game_factory(game_type):
    if game_type == "sudoku":
        from .sudoku import SudokuPuzzleFactory as GameFactory
    elif game_type == "binairo":
        from .binairo import BinairoPuzzleFactory as GameFactory
    elif game_type == "coloredsudoku":
        from .coloredsudoku import ColoredSudokuPuzzleFactory as GameFactory
    elif game_type == "kakuro":
        from .kakuro import KakuroPuzzleFactory as GameFactory
    elif game_type == "killersudoku":
        from .killersudoku import KillerSudokuPuzzleFactory as GameFactory
    elif game_type == "renzoku":
        from .renzoku import RenzokuPuzzleFactory as GameFactory
    elif game_type == "skyscraper":
        from .skyscraper import SkyscraperPuzzleFactory as GameFactory
    elif game_type == "starbattle":
        from .starbattle import StarBattlePuzzleFactory as GameFactory
    elif game_type == "treesandtents":
        from .treesandtents import TreesAndTentsPuzzleFactory as GameFactory
    elif game_type == "thermometers":
        from .thermometers import ThermometersPuzzleFactory as GameFactory
    elif game_type == "futoshiki":
        from .futoshiki import FutoshikiPuzzleFactory as GameFactory
    elif game_type == "hitori":
        from .hitori import HitoriPuzzleFactory as GameFactory
    elif game_type == "aquarium":
        from .aquarium import AquariumPuzzleFactory as GameFactory
    elif game_type == "kakurasu":
        from .kakurasu import KakurasuPuzzleFactory as GameFactory
    elif game_type == "oddevensudoku":
        from .oddevensudoku import OddEvenSudokuPuzzleFactory as GameFactory
    elif game_type == "battleships":
        from .battleships import BattleshipsPuzzleFactory as GameFactory
    elif game_type == "fieldexplore":
        from .fieldexplore import FieldExplorePuzzleFactory as GameFactory
    elif game_type == "jigsawsudoku":
        from .jigsawsudoku import JigsawSudokuPuzzleFactory as GameFactory
    elif game_type == "lightup":
        from .lightup import LightUpPuzzleFactory as GameFactory
    elif game_type == "nonogram":
        from .nonogram import NonogramPuzzleFactory as GameFactory

    return GameFactory
