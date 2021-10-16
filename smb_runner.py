import random
import retro
import cv2
import numpy as np


class Runner:
    env = retro.RetroEnv
    # TODO: check if theres a better system to detect when a level is finished
    level_max_distances = {
        '1_1': 3175
    }
    addr_world = int('0x075F', 16)
    addr_level = int('0x0760', 16)
    # address of current tileset, containing 2 13*16 tilesets
    addr_tiles = [int('0x0500', 16), int('0x069F', 16)]
    # 1 if inside y-viewport, 0 if above, 2+ when falling down a pit
    addr_player_viewport_ypos = int('0x00B5', 16)
    # precise player y-position (0-255)
    addr_player_screen_ypos = int('0x03B8', 16)
    # player x-pos on screen
    addr_curr_x = int('0x0086', 16)
    # tileset containing the player
    addr_curr_page = int('0x006D', 16)
    # check if enemies are drawn, up to 5 at the same time
    addr_enemies_drawn = int('0x000F', 16)
    # enemy-positions
    addr_enemy_xpos_level = int('0x006E', 16)
    addr_enemy_xpos = int('0x0087', 16)
    addr_enemy_ypos = int('0x00CF', 16)
    # player-state, 6 & 11 mean dying and dead
    addr_player_state = int('0x000E', 16)
    render_ai_viewport = False
    render_env = False
    stages = [
        'Level1-1',
        'Level2-1',
        'Level3-1',
    ]

    def __init__(self):
        stage = random.choices(self.stages, weights=(25, 1, 1))
        self.env = retro.make(game='SuperMarioBros-Nes', obs_type=retro.Observations.RAM, state=stage[0])

    def run(self, action_activation_function):
        obs = self.env.reset()

        fitness = 0
        fitness_max = 0
        counter = 0
        xpos_max = 0
        done = False

        if self.render_ai_viewport:
            cv2.namedWindow("main", cv2.WINDOW_NORMAL)
            img = np.zeros((12 * 8, 11 * 8, 3), np.uint8)

        while not done:
            inputs = self.generate_ai_viewport(obs)

            if self.render_env:
                self.env.render()

            if self.render_ai_viewport:
                # render opencv grid
                for i, row in enumerate(inputs):
                    for j, tile in enumerate(row):
                        end_x = int(j * 8 + 8)
                        end_y = int(i * 8 + 8)
                        color = (0, 0, int(tile) * 255)
                        if tile == 2:
                            color = (0, 255, 0)
                        if tile == 3:
                            color = (255, 0, 0)
                        cv2.rectangle(img, (int(j * 8), int(i * 8)), (end_x, end_y), color, -1)
                cv2.imshow('main', img)
                cv2.waitKey(1)

            actions = action_activation_function(np.ndarray.flatten(inputs))
            # actions = [(i > 0) * i for i in actions]
            # insert missing inputs
            actions[1:1] = [.0, .0, .0, .0, .0]

            # Ich hab keine Ahnung wieso, aber mit dieser Änderung lernt die KI 1.000.000x schneller
            # evtl. haben Tweaks in der NEAT-Config einen ähnlichen Effekt
            actions = [int((i > .5)) for i in actions]

            obs, rew, done, info = self.env.step(actions)

            xpos = int(obs[self.addr_curr_page]) * 256 + int(obs[self.addr_curr_x])

            if xpos > xpos_max:
                if xpos >= 3175 and self.env.statename == 'Level1-1':
                    fitness_max = 5000
                    done = True
                    continue
                fitness = xpos
                xpos_max = xpos

            if fitness > fitness_max:
                fitness_max = fitness
                counter = 0
            else:
                counter += 1

            # check if player did not progress for 125 steps or died
            if obs[self.addr_player_state] in {6, 11} or counter > 125 or obs[self.addr_player_viewport_ypos] > 1:
                done = True

        if self.render_env:
            self.env.render(close=True)

        if self.render_ai_viewport:
            cv2.waitKey(1)
            cv2.destroyWindow('main')

        self.env.close()
        return fitness_max

    def generate_ai_viewport(self, obs: np.ndarray) -> np.ndarray:
        """
        Generates the input viewport for the ai calculated from the RAM-observation-space
        """
        viewport = np.ndarray((12, 11))
        tileset = np.array([obs[int(self.addr_tiles[0]): int(self.addr_tiles[1]) + 1]])
        tileset[tileset > 0] = 1
        tileset = np.reshape(
            # [int((i > 1)) for i in obs[int(self.addr_tiles[0]): int(self.addr_tiles[1]) + 1]], (26, 16)
            tileset, (26, 16)
        )
        tile_screen = np.concatenate((tileset[:13], tileset[13:]), axis=1)
        player_tile_xpos = int(obs[self.addr_curr_page] % 2) * 16 + int(obs[self.addr_curr_x]) // 16
        player_tile_ypos = (int(obs[self.addr_player_screen_ypos]) - 1) // 16

        for i, enemy in enumerate(obs[self.addr_enemies_drawn:self.addr_enemies_drawn + 5]):
            # skip if not an enemy or enemy-y-position is greater than 13 tiles
            if enemy == 0 or obs[self.addr_enemy_ypos + i] > 223:
                continue
            tile_screen[
                (int(obs[self.addr_enemy_ypos + i])) // 16 - 1
            ][
                int(obs[self.addr_enemy_xpos_level + i]) % 2 * 16 + int(obs[self.addr_enemy_xpos + i]) // 16
            ] = 3

        # fill ai_viewport
        for i, row in enumerate(tile_screen):
            if i == player_tile_ypos:
                # TODO: check how player-position in new grid should be calculated
                row[player_tile_xpos] = 2
            if i == 12:
                continue
            col_left = player_tile_xpos - 2
            j = 0
            while col_left < 11 + player_tile_xpos - 2:
                viewport[i][j] = row[col_left % 32]
                col_left = col_left + 1
                j = j + 1
        return viewport
