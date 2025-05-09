package parseptron

fun <T> List<List<T>>.transposition(): List<List<T>> {
    val t = mutableListOf<MutableList<T>>()

    for (w in this[0].indices) {
        val column = mutableListOf<T>()

        for (h in this.indices) {
            column.add(this[h][w])
        }
        t.add(column)
    }

    return t
}

fun <T> Array<Array<T>>.transposition(): MutableList<MutableList<T>> {
    val t = mutableListOf<MutableList<T>>()

    for (w in this[0].indices) {
        val column = mutableListOf<T>()

        for (h in this.indices) {
            column.add(this[h][w])
        }
        t.add(column)
    }

    return t
}

fun matrixMultiplication(
    leftMatrix: List<List<Double>>,
    rightMatrix: List<Double>,
): List<Double> {
    val result = mutableListOf<Double>()

    if (leftMatrix[0].count() != rightMatrix.count()) {
        throw IllegalArgumentException("Ширина левой матрицы должна совпадать с высотой правой матрицы для успешного умножения. Ширина левой матрицы ${leftMatrix[0].count()}, высота правой ${rightMatrix.count()}")
    } else {

        leftMatrix.forEachIndexed { height, row ->
            var sum = 0.0
            row.forEachIndexed { width, value ->

                sum += (value * rightMatrix[width])
            }

            result.add(sum)
        }
    }

    return result
}

fun List<List<Double>>.elementWiseMultiplication(value: Double): List<List<Double>> {
    val result = mutableListOf<MutableList<Double>>()

    this.forEach { row ->
        val newRow = mutableListOf<Double>()
        row.forEach {
            newRow.add(it * value)
        }
        result.add(newRow)
    }

    return result
}

fun List<Double>.vectorWiseMultiplication(vector: List<Double>): List<Double> {
    val result = mutableListOf<Double>()

    if (this.count() == vector.count()) {
        this.forEachIndexed { index, value ->
            result.add(value * vector[index])
        }
    }else {
        throw IllegalArgumentException("Выктора имеют разную высоту")
    }

    return result
}