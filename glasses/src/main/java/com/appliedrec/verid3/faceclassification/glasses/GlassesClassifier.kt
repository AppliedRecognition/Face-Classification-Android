package com.appliedrec.verid3.faceclassification.glasses

import android.content.Context
import com.appliedrec.verid3.faceclassification.FaceClassifier

class GlassesClassifier private constructor(nativeContext: Long) :
    FaceClassifier<GlassesClassificationResult>(nativeContext) {

    override fun transformResult(result: FloatArray): GlassesClassificationResult {
        require(result.size >= 2)
        return GlassesClassificationResult(result[0], result[1])
    }

    companion object {
        @JvmStatic
        suspend fun create(context: Context): GlassesClassifier {
            val modelFileName = "glasses-20231216na-ffv2702_fft4539_clbatl3641_detface94_detface18-mutate-mirror-rgb-yolo100-log-r12s4000-q08.nv"
            val nativeContext = FaceClassifier.createContext(
                context,
                "Glasses",
                modelFileName
            )
            return GlassesClassifier(nativeContext)
        }
    }
}