package com.appliedrec.verid3.faceclassification.covering

import android.content.Context
import com.appliedrec.verid3.faceclassification.FaceClassifier

class FaceCoveringClassifier private constructor(nativeContext: Long) :
    FaceClassifier<Float>(nativeContext) {

    override fun transformResult(result: FloatArray): Float {
        require(result.isNotEmpty())
        return result[0]
    }

    companion object {
        @JvmStatic
        suspend fun create(context: Context): FaceCoveringClassifier {
            val modelFileName = "facemask-20200720-dut23td+1700-251149%3000.nv"
            val nativeContext = FaceClassifier.createContext(
                context,
                "FaceCovering",
                modelFileName
            )
            return FaceCoveringClassifier(nativeContext)
        }
    }
}