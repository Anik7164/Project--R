from click_balls import ball_box


def test_click_snaps_to_a_guess_or_sizes_from_the_nearest_player():
    guesses = [((100, 100, 110, 110), 0.2)]
    assert ball_box(112, 104, guesses, [(0, (0, 0, 20, 100))]) == (100, 100, 110, 110)
    x1, y1, x2, y2 = ball_box(500, 300, guesses, [(0, (480, 220, 500, 300)),
                                                   (0, (0, 0, 40, 400))])
    assert abs((x2 - x1) - 0.12 * 80) < 1e-6 and (x1 + x2) / 2 == 500
