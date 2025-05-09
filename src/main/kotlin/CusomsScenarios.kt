import parseptron.Parceptron2
import parseptron.Parseptron
import java.io.File
import javax.imageio.ImageIO
import kotlin.random.Random

fun learningApplyAvarageGradientWithIncreasingAccuracy() {

    val parceptron = Parseptron()

    val examplesCount = 2000

    var p = 1
    while (p < 2000) {

        println("learning progress $p from $examplesCount")

        repeat(100) {
            val i = Random.nextInt(10)
            val image =
                ImageIO.read(File("D:\\IntelliJ\\parceptron-data-set\\dataset\\MNIST - JPG - training\\$i\\p (${p + it}).jpg"))

            parceptron.learning(image, i)

        }
        parceptron.applyAllGradients()
        p += 100
    }


    testing(parceptron)
}

fun testing(
    parceptron: Parseptron
) {
    var testingErrorCount = 0
    for (i in 0..9) {
        for (p in 1..1) {

            val image =
                ImageIO.read(File("D:\\IntelliJ\\parceptron-data-set\\dataset\\MNIST - JPG - testing\\$i\\p ($p).jpg"))

            val result = parceptron.recognize(image)

            println("Ожилания: $i, Результат: $result")

            println(parceptron.outputLayer.toList())
            if (result != i) {

                testingErrorCount++
            }
        }
    }

    println(testingErrorCount)
}

fun learningSumOfNumbers(): Parseptron {

    val parceptron = Parseptron(
        inputSize = 2,
        firstLayerSize = 4,
        secondLayerSize = 4,
        outputSize = 2
    )

    val timeRepeat = 10

    repeat(timeRepeat) {
        sumDataSet.forEach {
            parceptron.learning(it.first.first, it.first.second, it.second)
        }
        parceptron.applyAllGradients()
    }

    return parceptron
}


fun learningParceptron2() {

    val parceptron = Parceptron2()

    val examplesCount = 2000

    var p = 1
    while (p < 2000) {

        println("learning progress $p from $examplesCount")

        repeat(100) {
            val i = Random.nextInt(10)
            val image =
                ImageIO.read(File("D:\\IntelliJ\\parceptron-data-set\\dataset\\MNIST - JPG - training\\$i\\p (${p + it}).jpg"))

            parceptron.learning(image, i)
        }

        parceptron.applyAllGradients()
        p += 100
    }

    testing2(parceptron)
}

fun learningParceptron2V2() {

    val parceptron = Parceptron2()

    val examplesCount = 500

    repeat(10) {
        for (p in 1 until examplesCount) {
            println("learning progress $p from $examplesCount")

            for (s in 0..9) {
                val image =
                    ImageIO.read(File("D:\\IntelliJ\\parceptron-data-set\\dataset\\MNIST - JPG - training\\$s\\p ($p).jpg"))

                parceptron.learning(image, s)
            }

            parceptron.applyAllGradients()
        }
    }

    testing2(parceptron)
}

fun testing2(
    parceptron: Parceptron2
) {
    var testingErrorCount = 0
    for (i in 0..9) {
        for (p in 1..1) {

            val image =
                ImageIO.read(File("D:\\IntelliJ\\parceptron-data-set\\dataset\\MNIST - JPG - testing\\$i\\p ($p).jpg"))

            val result = parceptron.recognize(image)

            println("Ожилания: $i, Результат: $result")
            println(parceptron.outputLayer.toList())

            if (result != i) {

                testingErrorCount++
            }
        }
    }

    println(testingErrorCount)
}