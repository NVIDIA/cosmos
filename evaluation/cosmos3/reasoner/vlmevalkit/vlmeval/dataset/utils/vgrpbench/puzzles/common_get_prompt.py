def get_prompt(game_type: str, thinking_format: str) -> str:
    if game_type == "sudoku":
        from puzzles.sudoku import SYSTEM_PROMPT
    elif game_type == "coloredsudoku":
        from puzzles.coloredsudoku import SYSTEM_PROMPT
    elif game_type == "binairo":
        from puzzles.binairo import SYSTEM_PROMPT
    elif game_type == "futoshiki":
        from puzzles.futoshiki import SYSTEM_PROMPT
    elif game_type == "hitori":
        from puzzles.hitori import SYSTEM_PROMPT
    elif game_type == "kakuro":
        from puzzles.kakuro import SYSTEM_PROMPT
    elif game_type == "killersudoku":
        from puzzles.killersudoku import SYSTEM_PROMPT
    elif game_type == "renzoku":
        from puzzles.renzoku import SYSTEM_PROMPT
    elif game_type == "skyscraper":
        from puzzles.skyscraper import SYSTEM_PROMPT
    elif game_type == "starbattle":
        from puzzles.starbattle import SYSTEM_PROMPT
    elif game_type == "sudoku":
        from puzzles.sudoku import SYSTEM_PROMPT
    elif game_type == "treesandtents":
        from puzzles.treesandtents import SYSTEM_PROMPT
    elif game_type == "thermometers":
        from puzzles.thermometers import SYSTEM_PROMPT
    elif game_type == "kakurasu":
        from puzzles.kakurasu import SYSTEM_PROMPT
    elif game_type == "aquarium":
        from puzzles.aquarium import SYSTEM_PROMPT
    elif game_type == "oddevensudoku":
        from puzzles.oddevensudoku import SYSTEM_PROMPT

    elif game_type == "battleships":
        from puzzles.battleships import SYSTEM_PROMPT
    elif game_type == "fieldexplore":
        from puzzles.fieldexplore import SYSTEM_PROMPT
    elif game_type == "jigsawsudoku":
        from puzzles.jigsawsudoku import SYSTEM_PROMPT
    elif game_type == "nonogram":
        from puzzles.nonogram import SYSTEM_PROMPT
    elif game_type == "lightup":
        from puzzles.lightup import SYSTEM_PROMPT

    else:
        raise ValueError(f"Unknown game type: {game_type}")

    if thinking_format == "direct_solution":
        return SYSTEM_PROMPT["direct_solution"]
    else:
        return SYSTEM_PROMPT["cot"]
