import retro
import numpy as np
import cv2

# - TODO: evaluate if writing memory locations into data.json and reading from info object is faster than reading from
#    RAM array

env = retro.make(game='SuperMarioBros-Nes', obs_type=retro.Observations.RAM)
addr_tiles = [int('0x0500', 16), int('0x069F', 16)]
addr_player_screen_xpos = int('0x03AD', 16)
addr_player_screen_ypos = int('0x03B8', 16)
addr_xpos_low = int('0x071C', 16)
addr_xpos_high = int('0x071A', 16)
addr_curr_x = int('0x0086', 16)
addr_curr_page = int('0x006D', 16)
addr_enemies_drawn = int('0x000F', 16)
addr_enemy_xpos_level = int('0x006E', 16)
addr_enemy_xpos = int('0x0087', 16)
addr_enemy_ypos = int('0x00CF', 16)


def main():
    env.reset()
    env.render()

    img = np.zeros((12 * 8, 11 * 8, 3), np.uint8)
    cv2.namedWindow("main", cv2.WINDOW_NORMAL)

    done = False
    ai_viewport = np.zeros((12, 11))
    while not done:
        actions = env.action_space.sample()
        actions[7] = 1
        obs, rew, done, info = env.step(actions)
        env.render()

        tile_split = np.reshape(
            [int((i > 1)) for i in obs[int(addr_tiles[0]): int(addr_tiles[1]) + 1]], (26, 16)
        )
        tile_screen = np.concatenate((tile_split[:13], tile_split[13:]), axis=1)

        player_tile_xpos = int(obs[addr_curr_page] % 2) * 16 + int(obs[addr_curr_x]) // 16
        player_tile_ypos = (int(obs[addr_player_screen_ypos]) - 1) // 16

        for i, enemy in enumerate(obs[addr_enemies_drawn:addr_enemies_drawn + 5]):
            if enemy != 0:
                tile_screen[
                    (int(obs[addr_enemy_ypos + i])) // 16 - 1
                ][
                    int(obs[addr_enemy_xpos_level + i]) % 2 * 16 + int(obs[addr_enemy_xpos + i]) // 16
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
                ai_viewport[i][j] = row[col_left % 32]
                col_left = col_left + 1
                j = j + 1

        # render opencv grid
        for i, row in enumerate(ai_viewport):
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

    env.reset()


if __name__ == "__main__":
    main()
