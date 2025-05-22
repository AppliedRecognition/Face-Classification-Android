package com.appliedrec.verid3.faceclassification.idcard

import android.graphics.BitmapFactory
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.appliedrec.verid3.common.Image
import com.appliedrec.verid3.common.serialization.fromBitmap
import com.appliedrec.verid3.facedetection.mplandmarks.FaceDetection
import kotlinx.coroutines.runBlocking

import org.junit.Test
import org.junit.runner.RunWith

import org.junit.Assert.*

/**
 * Instrumented test, which will execute on an Android device.
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
@RunWith(AndroidJUnit4::class)
class FaceCoveringClassifierTest {

    @Test
    fun testCreateClassifier() = runBlocking {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val classifier = FaceCoveringClassifier.create(appContext)
        assertNotNull(classifier)
    }

    @Test
    fun testDetectFaceCovering() = runBlocking {
        val result = runDetection("face-covering.jpg")
        assertTrue(result > 0.5)
    }

    @Test
    fun testDetectNoFaceCovering() = runBlocking {
        val result = runDetection("no-face-covering.jpg")
        assertTrue(result < 0.5)
    }

    private suspend fun runDetection(imageName: String): Float {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val classifier = FaceCoveringClassifier.create(appContext)
        val faceDetection = FaceDetection(appContext)
        val image = appContext.assets.open(imageName).use {
            val bitmap = BitmapFactory.decodeStream(it)
            Image.fromBitmap(bitmap)
        }
        val face = faceDetection.detectFacesInImage(image, 1).first()
        return classifier.use { it.extractAttribute(face, image) }
    }
}