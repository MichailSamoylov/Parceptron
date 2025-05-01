import androidx.compose.ui.window.application
import parseptron.Parseptron
import java.io.File
import javax.imageio.ImageIO


fun main() = application {

    //learningAllImage()
    learningAllImage3()

//    val parceptron = Parseptron()
//    visual(parceptron)
}

fun learningInt() {

//    val parceptron = Parseptron()
//
//    lineNeuronDataSet.forEach {
//        parceptron.learning(it.first, it.second)
//        println(parceptron.outputLayer.toList())
//    }
//
//    lineNeuronDataSet.forEach {
//        println(parceptron.recognize(it.first))
//    }
}

fun learningOneSymbolImage(symbol: Int, parceptron: Parseptron) {

    for (p in 1..1000) {

        val image = ImageIO.read(File("D:\\IntelliJ\\parseptron\\dataset\\MNIST - JPG - training\\$symbol\\p ($p).jpg"))
        parceptron.learning(image, symbol)

        println("learning symbol $symbol example $p from 1000")

        parceptron.applyAllGradients()
    }

    val image = ImageIO.read(File("D:\\IntelliJ\\parseptron\\dataset\\MNIST - JPG - testing\\$symbol\\p (50).jpg"))
    println(parceptron.recognize(image))

    val imageNotSymbol = ImageIO.read(File("D:\\IntelliJ\\parseptron\\dataset\\MNIST - JPG - testing\\3\\p (1).jpg"))
    println(parceptron.recognize(imageNotSymbol))

    println(parceptron.outputLayer.toList())
}

fun learningAllImage() {

    val parceptron = Parseptron()

    val examplesCount = 2000

    for (p in 1..examplesCount) {

        for (i in 0..9) {

            val image = ImageIO.read(File("D:\\IntelliJ\\parseptron\\dataset\\MNIST - JPG - training\\$i\\p ($p).jpg"))

            parceptron.learning(image, i)

            //println("learning symbol $i example $p from $examplesCount")
        }

        println("learning symbols, example $p from $examplesCount")
        parceptron.applyAllGradients()

        for (i in 0..9) {

            val image = ImageIO.read(File("D:\\IntelliJ\\parseptron\\dataset\\MNIST - JPG - testing\\$i\\p (1).jpg"))


            val result = parceptron.recognize(image)
            println("testing symbol $i. Result - ${result}. Outputs ${parceptron.outputLayer.toList()}")
        }
    }

    var testingErrorCount = 0
    for (p in 1..890) {

        for (i in 0..9) {

            val image = ImageIO.read(File("D:\\IntelliJ\\parseptron\\dataset\\MNIST - JPG - testing\\$i\\p ($p).jpg"))

            println("testing symbol $i example $p from 890")
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

fun learningAllImage3() {

    val parceptron = Parseptron()

    val examplesCount = 2000
    for (i in 0..9) {

        for (p in 1..examplesCount) {

            if(p == 1){
                println("learning symbol $i")
            }

            val image = ImageIO.read(File("D:\\IntelliJ\\parseptron\\dataset\\MNIST - JPG - training\\$i\\p ($p).jpg"))

            parceptron.learning(image, i)
            parceptron.applyAllGradients()

            val imageTest = ImageIO.read(File("D:\\IntelliJ\\parseptron\\dataset\\MNIST - JPG - testing\\$i\\p (1).jpg"))
            val result = parceptron.recognize(imageTest)


            val secondCondition = parceptron.outputLayer.toList().filter {
                if (it == parceptron.outputLayer.max()) {
                    false
                } else {
                    true
                }
            }.none {
                parceptron.outputLayer.max() / 100 < it
            }

            if (result == i && secondCondition) {
                break
            }
        }

        println("Обучение для $i завершено ${parceptron.outputLayer.toList()}")
    }

    var testingErrorCount = 0
    for (p in 1..890) {

        for (i in 0..9) {

            val image = ImageIO.read(File("D:\\IntelliJ\\parseptron\\dataset\\MNIST - JPG - testing\\$i\\p ($p).jpg"))


            val result = parceptron.recognize(image)
            println(result)
            println("testing symbol $i example $p from 890")

            if (result != i) {

                testingErrorCount++
            }
        }
    }

    println(testingErrorCount)

    println(parceptron.outputLayer.toList())
}
