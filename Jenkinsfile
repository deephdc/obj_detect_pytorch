#!/usr/bin/groovy

@Library(['github.com/indigo-dc/jenkins-pipeline-library@release/1.4.0']) _

def job_result_url = ''

ci_cd_image = 'deephdc/ci_cd-obj_detect_pytorch'
if (env.BRANCH_NAME == 'test') {
    ci_cd_image = 'deephdc/ci_cd-obj_detect_pytorch:test'
}

pipeline {
    agent {
        //label 'python3.6'
        docker { image "${ci_cd_image}" }
    }

    environment {
        author_name = "Silke Donayre"
        author_email = "silke.holtz@student.kit.edu"
        app_name = "obj_detect_pytorch"
        job_location = "Pipeline-as-code/DEEP-OC-org/DEEP-OC-obj_detect_pytorch/${env.BRANCH_NAME}"
    }

    stages {
        stage('Code fetching') {
            steps {
                checkout scm
            }
        }

        stage('Style analysis') {
            steps {
                ToxEnvRun('pep8')
            }
            post {
                always {
                    WarningsReport('Pep8')
                }
            }
        }

        stage('Unit testing coverage') {
            steps {
                ToxEnvRun('cover')
                ToxEnvRun('cobertura')
            }
            post {
                success {
                    HTMLReport('cover', 'index.html', 'coverage.py report')
                    CoberturaReport('**/coverage.xml')
                }
            }
        }

        stage('Metrics gathering') {
            steps {
                checkout scm
                SLOCRun()
            }
            post {
                success {
                    SLOCPublish()
                }
            }
        }

        stage('Security scanner') {
            steps {
                ToxEnvRun('bandit-report')
                script {
                    if (currentBuild.result == 'FAILURE') {
                        currentBuild.result = 'UNSTABLE'
                    }
              }
            }
            post {
               always {
                    HTMLReport("tmp/bandit", 'index.html', 'Bandit report')
                }
            }
        }

        stage("Re-build Docker images") {
            when {
                anyOf {
                   branch 'master'
                   branch 'test'
                   buildingTag()
               }
            }
            steps {
                script {
                    def job_result = JenkinsBuildJob("${env.job_location}")
                    job_result_url = job_result.absoluteUrl
                }
            }
        }

    }

    post {
        failure {
            script {
                currentBuild.result = 'FAILURE'
            }
        }

        always  {
            script { //stage("Email notification")
                def build_status =  currentBuild.result
                build_status =  build_status ?: 'SUCCESS'
                def subject = """
New ${app_name} build in Jenkins@DEEP:\
${build_status}: Job '${env.JOB_NAME}\
[${env.BUILD_NUMBER}]'"""

                def body = """
Dear ${author_name},\n\n
A new build of '${app_name} (${env.BRANCH_NAME})' DEEP application is available in Jenkins at:\n\n
*  ${env.BUILD_URL}\n\n
terminated with '${build_status}' status.\n\n
Check console output at:\n\n
*  ${env.BUILD_URL}/console\n\n
and resultant Docker image rebuilding job at (may be empty in case of FAILURE):\n\n
*  ${job_result_url}\n\n

DEEP Jenkins CI service"""

                EmailSend(subject, body, "${author_email}")
            }
        }
    }
}
