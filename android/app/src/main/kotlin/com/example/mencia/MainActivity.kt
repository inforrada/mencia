package com.example.mencia

import android.app.Activity
import android.content.Intent
import android.net.Uri
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.io.FileOutputStream

class MainActivity : FlutterActivity() {
    private val channelName = "com.example.mencia/image_picker"
    private val pickImageRequestCode = 9401
    private var pendingResult: MethodChannel.Result? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, channelName)
            .setMethodCallHandler { call, result ->
                if (call.method == "pickExistingImage") {
                    if (pendingResult != null) {
                        result.error("ALREADY_ACTIVE", "Image picker is already active.", null)
                        return@setMethodCallHandler
                    }
                    pendingResult = result
                    val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                        addCategory(Intent.CATEGORY_OPENABLE)
                        type = "image/*"
                    }
                    startActivityForResult(intent, pickImageRequestCode)
                } else {
                    result.notImplemented()
                }
            }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == pickImageRequestCode) {
            val callback = pendingResult
            pendingResult = null
            if (callback == null) return

            if (resultCode == Activity.RESULT_OK && data != null) {
                val uri: Uri? = data.data
                if (uri != null) {
                    try {
                        val tempFile = copyUriToTempFile(uri)
                        callback.success(tempFile.absolutePath)
                    } catch (e: Exception) {
                        callback.error("COPY_FAILED", "Failed to read selected image: ${e.message}", null)
                    }
                } else {
                    callback.success(null)
                }
            } else {
                callback.success(null)
            }
        }
    }

    private fun copyUriToTempFile(uri: Uri): File {
        val tempFile = File(cacheDir, "picked_mood_ref_${System.currentTimeMillis()}.jpg")
        contentResolver.openInputStream(uri)?.use { input ->
            FileOutputStream(tempFile).use { output ->
                input.copyTo(output)
            }
        } ?: throw IllegalStateException("Unable to open input stream for URI: $uri")
        return tempFile
    }
}


