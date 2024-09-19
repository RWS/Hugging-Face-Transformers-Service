using System;
using System.IO;
using System.Text;
using Microsoft.Deployment.WindowsInstaller;

namespace HuggingFaceTS.ConfigurationTasks
{
    public class CustomActions
	{
		[CustomAction]
		public static ActionResult SaveUserConfiguration(Session session)
        {
            // Retrieve configuration settings from session properties
            var installDir = session["INSTALLFOLDER"];
            var modelsDir = session["HUGGINGFACE_MODELS_DIR"];
            var token = session["HUGGINGFACE_TOKEN"];
            var host = session["HOST"];
            var port = session["PORT"];

            // Validate the installation directory
            if (string.IsNullOrEmpty(installDir) || !Directory.Exists(installDir))
            {
                session.Log("Installation directory does not exist: " + installDir);
                return ActionResult.NotExecuted;
            }

            var envFilePath = Path.Combine(installDir, ".env");

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
                    .AppendLine("# HOST IP address to run the REST API application")
                    .AppendLine($"HOST={host}")
                    .AppendLine()
                    .AppendLine("# Port number to run the REST API application")
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
