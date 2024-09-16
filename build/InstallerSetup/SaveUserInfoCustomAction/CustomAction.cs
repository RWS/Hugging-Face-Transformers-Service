using System;
using System.IO;
using System.Text;
using Microsoft.Deployment.WindowsInstaller;

namespace SaveUserInfoCustomAction
{
    public class CustomActions
    {
        [CustomAction]
        public static ActionResult SaveUserInfo(Session session)
        {
            // Retrieve configuration settings from session properties
            string installDir = session["INSTALLFOLDER"];
            string modelsDir = session["HUGGINGFACE_MODELS_DIR"];
            string token = session["HUGGINGFACE_TOKEN"];
            string port = session["PORT"];

            // Validate the installation directory
            if (string.IsNullOrEmpty(installDir) || !Directory.Exists(installDir))
            {
                session.Log("Installation directory does not exist: " + installDir);
                return ActionResult.NotExecuted;
            }

            string envFilePath = Path.Combine(installDir, ".env");

            try
            {
                // Construct environment file content using string interpolation
                var envFileContent = new StringBuilder()
                    .AppendLine("# Directory where Hugging Face models will be cached")
                    .AppendLine($"HUGGINGFACE_MODELS_DIR={modelsDir}")
                    .AppendLine()
                    .AppendLine("# Your Hugging Face API token for authentication")
                    .AppendLine($"HUGGINGFACE_TOKEN={token}")
                    .AppendLine()
                    .AppendLine("# Port to run the FastAPI application")
                    .AppendLine($"PORT={port}")
                    .ToString();

                // Write the content to the .env file
                File.WriteAllText(envFilePath, envFileContent, Encoding.UTF8);

                session.Log(".env file created successfully at: " + envFilePath);
                return ActionResult.Success;
            }
            catch (Exception ex)
            {
                // Log the exception and return a failure result
                session.Log("Error creating .env file: " + ex.Message);
                return ActionResult.Failure;
            }
        }
    }
}
