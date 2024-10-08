ENEMY_GROUP_1 = [3, 4, 5]
ENEMY_GROUP_2 = [1, 2, 6, 7, 8]
ALL_ENEMIES = [1, 2, 3, 4, 5, 6, 7, 8]


def enemy_group_to_str(arr: list):
    return "_".join([str(x) for x in arr])
