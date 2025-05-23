pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        maven {
            url = uri("https://maven.pkg.github.com/AppliedRecognition/Ver-ID-3D-Android-Libraries")
            credentials {
                username = settings.extra["gpr.user"] as String?
                password = settings.extra["gpr.token"] as String?
            }
        }
        mavenLocal()
    }
}

rootProject.name = "Face Classification"
include(":face-classification-core")
include(":face-classification-face-covering")
include(":face-classification-glasses")
include(":face-classification-id-card-face")