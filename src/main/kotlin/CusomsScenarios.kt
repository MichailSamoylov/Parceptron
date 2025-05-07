import parseptron.Parseptron
import java.io.File
import javax.imageio.ImageIO

fun learningOneSymbolImage(symbol: Int, parceptron: Parseptron) {

    for (p in 1..1000) {

        val image = ImageIO.read(File("D:\\IntelliJ\\parceptron-data-set\\dataset\\MNIST - JPG - training\\$symbol\\p ($p).jpg"))
        parceptron.learning(image, symbol)

        println("learning symbol $symbol example $p from 1000")

        parceptron.applyAllGradients()
    }

    testing(parceptron)
}

fun learningImageApplyGradientEachStepWithIncreasingAccuracy() {

    val parceptron = Parseptron()

    val examplesCount = 2000
    val rounds = 3

    repeat(rounds) { round ->
        for (i in 0..9) {

            for (p in 1..examplesCount) {

                if (p == 1) {
                    println("learning symbol $i")
                }

                val image =
                    ImageIO.read(File("D:\\IntelliJ\\parceptron-data-set\\dataset\\MNIST - JPG - training\\$i\\p ($p).jpg"))

                parceptron.learning(image, i)
                parceptron.applyAllGradients()

                val imageTest =
                    ImageIO.read(File("D:\\IntelliJ\\parceptron-data-set\\dataset\\MNIST - JPG - testing\\$i\\p (1).jpg"))
                val result = parceptron.recognize(imageTest)

                val secondCondition = parceptron.outputLayer.toList().filter {
                    if (it == parceptron.outputLayer.max()) {
                        false
                    } else {
                        true
                    }
                }.none {
                    parceptron.outputLayer.max() / (10 * round) < it
                } && parceptron.outputLayer.max() > 0.8

                if (result == i && secondCondition) {
                    println("break: $p")
                    break
                }
            }

            println("Обучение для $i завершено ${parceptron.outputLayer.toList()}")
        }
    }

    testing(parceptron)
}

fun learningApplyAvarageGradientWithIncreasingAccuracy() {

    val parceptron = Parseptron()

    val examplesCount = 2000
    val rounds = 5

    repeat(rounds) { round ->

        for (p in 1..examplesCount) {

            for (i in 0..9) {

                if (p == 1) {
                    println("learning example $p")
                }

                val image =
                    ImageIO.read(File("D:\\IntelliJ\\parceptron-data-set\\dataset\\MNIST - JPG - training\\$i\\p ($p).jpg"))

                parceptron.learning(image, i)
            }

            parceptron.applyAllGradients()

            for (i in 0..9) {
                val imageTest =
                    ImageIO.read(File("D:\\IntelliJ\\parceptron-data-set\\dataset\\MNIST - JPG - testing\\$i\\p (1).jpg"))

                val result = parceptron.recognize(imageTest)

                val secondCondition = parceptron.outputLayer.toList().filter {
                    if (it == parceptron.outputLayer.max()) {
                        false
                    } else {
                        true
                    }
                }.none {
                    parceptron.outputLayer.max() / 10 < it
                } && parceptron.outputLayer.max() > 0.8

                println("testing result: $i to $result && ${parceptron.outputLayer.toList()}")

                if (result == i && secondCondition) {
//                    println("break: $p")
//                    break
                }
            }
        }
    }

//    var testingErrorCount = 0
//    for (p in 1..320) {
//
//        println("testing example $p from 320")
//
//        for (i in 0..9) {
//
//            val image =
//                ImageIO.read(File("D:\\IntelliJ\\parceptron-data-set\\dataset\\MNIST - JPG - testing\\$i\\p ($p).jpg"))
//
//
//            val result = parceptron.recognize(image)
//            println(result)
//
//
//            if (result != i) {
//
//                testingErrorCount++
//            }
//        }
//    }
//
//    println(testingErrorCount)
//
//    println(parceptron.outputLayer.toList())
}

fun testing(
    parceptron: Parseptron,
    input: Int,
): Int {
    val image =
        ImageIO.read(File("D:\\IntelliJ\\parceptron-data-set\\dataset\\MNIST - JPG - testing\\$input\\p (1).jpg"))

    return parceptron.recognize(image)
}

fun learningAllImage1() {

    val parceptron = Parseptron()

    val examplesCount = 2000
    for (i in 0..9) {

        for (p in 1..examplesCount) {

            if (p == 1) {
                println("learning symbol $i")
            }

            val image =
                ImageIO.read(File("D:\\IntelliJ\\parceptron-data-set\\dataset\\MNIST - JPG - training\\$i\\p ($p).jpg"))

            parceptron.learning(image, i)
            parceptron.applyAllGradients()

            val imageTest =
                ImageIO.read(File("D:\\IntelliJ\\parceptron-data-set\\dataset\\MNIST - JPG - testing\\$i\\p (1).jpg"))
            val result = parceptron.recognize(imageTest)

            val secondCondition = parceptron.outputLayer.toList().filter {
                if (it == parceptron.outputLayer.max()) {
                    false
                } else {
                    true
                }
            }.none {
                parceptron.outputLayer.max() / 100 < it
            } && parceptron.outputLayer.max() > 0.8

            if (result == i && secondCondition) {
                break
            }
        }

        println("Обучение для $i завершено ${parceptron.outputLayer.toList()}")
    }

    var testingErrorCount = 0
    for (p in 1..320) {

        println("testing example $p from 320")

        for (i in 0..9) {

            val image =
                ImageIO.read(File("D:\\IntelliJ\\parceptron-data-set\\dataset\\MNIST - JPG - testing\\$i\\p ($p).jpg"))


            val result = parceptron.recognize(image)
            println(result)


            if (result != i) {

                testingErrorCount++
            }
        }
    }

    println(testingErrorCount)

    println(parceptron.outputLayer.toList())
}

fun learningAllImage2() {

    val parceptron = Parseptron()

    val examplesCount = 2000
    for (i in 0..9) {

        for (p in 1..examplesCount) {

            if (p == 1) {
                println("learning symbol $i")
            }

            val image =
                ImageIO.read(File("D:\\IntelliJ\\parceptron-data-set\\dataset\\MNIST - JPG - training\\$i\\p ($p).jpg"))

            parceptron.learning(image, i)

            parceptron.applyAllGradients()


            val (result, outputCondition) = getOutputCondition(parceptron, i)

            val previousCheckCondition = getPreviousCheckCondition(parceptron, i)

            if (!previousCheckCondition) {
                //Дообучить старые

            }

            if (result == i && outputCondition && previousCheckCondition) {
                break
            }
        }

        println("Обучение для $i завершено ${parceptron.outputLayer.toList()}")
    }

    var testingErrorCount = 0
    for (p in 1..320) {

        println("testing example $p from 320")

        for (i in 0..9) {

            val image =
                ImageIO.read(File("D:\\IntelliJ\\parceptron-data-set\\dataset\\MNIST - JPG - testing\\$i\\p ($p).jpg"))


            val result = parceptron.recognize(image)
            println(result)


            if (result != i) {

                testingErrorCount++
            }
        }
    }

    println(testingErrorCount)

    println(parceptron.outputLayer.toList())
}

fun getPreviousCheckCondition(
    parceptron: Parseptron,
    currentSymbol: Int
): Boolean {
    for (i in 0 until currentSymbol) {
        val (result, outputCondition) = getOutputCondition(parceptron, i)
    }

    return false
}

fun getOutputCondition(
    parceptron: Parseptron,
    input: Int
): Pair<Int, Boolean> {
    val imageTest =
        ImageIO.read(File("D:\\IntelliJ\\parceptron-data-set\\dataset\\MNIST - JPG - testing\\$input\\p (1).jpg"))
    val result = parceptron.recognize(imageTest)

    return result to (parceptron.outputLayer.toList().filter {
        if (it == parceptron.outputLayer.max()) {
            false
        } else {
            true
        }
    }.none {
        parceptron.outputLayer.max() / 100 < it
    } && parceptron.outputLayer.max() > 0.8)
}


private fun testing(
    parceptron: Parseptron
){

    var testingErrorCount = 0
    for (i in 0..9) {

        println("Тестирование цифры $i")

        for (p in 1..320) {

            val image = ImageIO.read(File("D:\\IntelliJ\\parceptron-data-set\\dataset\\MNIST - JPG - testing\\$i\\p ($p).jpg"))

            val result = parceptron.recognize(image)

            println("Ожилания: $i, Результат: $result")


            if (result != i) {

                testingErrorCount++
            }
        }
    }

    println(testingErrorCount)

    println(parceptron.outputLayer.toList())
}