import itertools

def win(current_game):

    def all_same(l):
        if l.count(l[0]) == len(l) and l[0]!=0 :
            return True
        else:
            return False

    #Horizontal Winner
    for row in current_game:
        if all_same(row):
            #print(row)
            print(f"Player {row[0]} is the winner horizontally!")
            return True

    #Vertical Win
    for col in range(len(current_game)):
        check = []
        for row in current_game:
            check.append(row[col])
        if all_same(check):
            #print(check)
            print(f"Player {check[0]} is the winner vertically!")
            return True

    #Diagonal Winner
    diag = []
    for ix in range(len(current_game)):
        diag.append(current_game[ix][ix])
    if all_same(diag):
        #print(diag) 
        print(f"Player {diag[0]} is the winner diagonally!") 
        return True
    for col, row in enumerate(reversed(range(len(current_game)))):
        diag.append(current_game[row][col])
    if all_same(diag):
        #print(diag2) 
        print(f"Player {diag[0]} is the winner diagonally!")
        return True

    return False

def game_board(game_map, player = 0, row = 0, column = 0, just_display = False):
    try:
        if game_map[row][column] != 0:
            print("This position is occupied")
            return game_map, False
        print("   "+"  ".join([str(i) for i in range(len(game_map))]))
        if just_display!= True:
            game_map[row][column] = player
        for count, row in enumerate (game_map):
            print(count, row)
        #print("\n")
        return game_map, True

    except IndexError as e:
        print("Make sure you input row/column as 0, 1 or 2 | Error:", e)
        return game_map, False

    except Exception as e:
        print ("Something went really wrong! | Error:", e)
        return game_map, False

'''
def win(current_game):
    for row in current_game:
        print(row)
        all_match = False
        i=0
        for item in row:
            if item == row[item]:
                i += 1
            if i == len(row):
                all_match = True
                print("Winner")         
'''
#win(game)

play = True
player = [1,2]

while play:
    game_size = int(input("What size game of tic tac toe? "))

    game = [[0 for i in range(game_size)] for i in range(game_size)]

    game_won = False
    game_board(game, just_display = True)

    player_choice = itertools.cycle(player)

    while not game_won:
        current_player = next(player_choice)
        print(f"Current player : {current_player}") 
        played = False

        while not played:
            row_choice  = int(input("\nWhat row do you want to play? (0, 1, 2...): "))
            column_choice  = int(input("What column do you want to play? (0, 1, 2...)\n: "))
            game, played = game_board(game, current_player, row_choice, column_choice)

        if win(game):
            game_won = True
            again = int(input("\nGame Over! Wanna play again? (1/0):"))
            if again == 1:
                print("\nRestarting the game")
            elif again == 0:
                print ("\nThanks! Bye")
                play = False
            else:
                print("\nNot a valid reponse. Leave")
                play = False
