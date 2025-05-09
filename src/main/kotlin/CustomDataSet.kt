/**
 * Дата сет для положительного или отрицательного числа
 * Первое число input
 * Второй ожидаемый результат
 * Ожидаемый результат 1 или 0 как boolean
 **/
val lineNeuronDataSet = listOf<Pair<Int, Int>>(
    -1 to 0,
    -2 to 0,
    -3 to 0,
    -4 to 0,
    -5 to 0,
    -6 to 0,
    -7 to 0,
    -8 to 0,
    -9 to 0,
    -10 to 0,
    0 to 1,
    1 to 1,
    2 to 1,
    3 to 1,
    4 to 1,
    5 to 1,
    6 to 1,
    7 to 1,
    8 to 1,
    9 to 1,
    10 to 1,
)

val sumDataSet = listOf<Pair<Pair<Int, Int>, Int>>(
    1 to 1 to 1,
    1 to 0 to 0,
    0 to 1 to 0,
    0 to 0 to 0,
)