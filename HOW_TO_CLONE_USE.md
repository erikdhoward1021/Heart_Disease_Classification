# User's Guide: How to Use the Standard Project Template for GSi Machine Learning Projects

Welcome to the user's guide for the Standard Project Template for Machine Learning Projects. This guide will walk you through the steps to clone, clean, and prepare a new repository using this template to jumpstart your machine learning projects. 

## Step 1: Clone the Template Repository
1. Create a new repository on GitHub for your machine learning project. Please make sure that the gsi-data-science team has Admin access to the repository.
2. When creating the new repository, select the option to create it from a template.
3. Choose Deloitte-US/GSi_SPT as the template for your new repository.
4. Clone the new repository to your local machine using the `git clone` command.

## Step 2: Customize the Template
1. Open the cloned repository in your preferred code editor.
2. Modify the existing files or add new files to fit your specific project needs.
3. Update the project name and description in the README.md file to reflect your project.
4. Remove any unnecessary files or folders that are not relevant to your project.

## Step 3: Clean and Prepare the Repository
1. Review the folder structure of the template and ensure it aligns with your project requirements.
2. Update the `/src` folder with your own source code files, including files with core algorithms, classes, modules, functions, and packages. For example, you might have modules here for data loading, preprocessing, model training, evaluation, and (potentially) serving.
3. Create corresponding unit tests for your source code files in the `/tests` folder.
4. Add documentation for your project in the `/docs` folder, including user guides and API documentation.
5. Place your raw data files in the `/data/raw` folder, ensuring they are unmodified and in their original format.
6. Clean and preprocess your data, and save the processed data in the `/data/processed` folder.
7. Use Jupyter notebooks for exploratory data analysis and model development, and store them in the `/notebooks` folder.
8. Save your trained machine learning models and related files in the `/models` folder.
9. Include any utility scripts in the `/scripts` folder for items like data preprocessing, model training, and/or utility scripts that automate certain tasks. These scripts are often more procedural (rather than modular) and task-oriented, and they might use the modules and functions defined in the src directory.
10. Store the output of your data analysis and model training, such as figures, tables, and performance metrics, in the `/results` folder.
11. Update the README often! (see below for a template)

## Step 4: Commit and Push Changes
1. Use the `git add <file>` command to stage the modified or new files.
2. Use the `git commit -m '<comment here>'` command to commit your changes with a descriptive commit message.
3. Use the `git push origin <branch_name>` command to push your changes to the remote repository on GitHub.

## Step 5: Collaborate and Iterate
1. Again, please ensure that the GitHub team 'gsi-data-science' has access to the repository.
2. Capture identifed features, bugs, enhancements, etc. using the Issues feature in GitHub
3. Use branching to develop those new features or fix bugs. Create a new branch using the `git checkout -b <branch_name>` command.
4. Make changes in your branch, commit them, and push them to the remote repository.
5. Submit a pull request to merge your changes into the main branch after reviewing, testing, and confirming the updates.
6. Collaborate with the team, gather feedback, and iterate on the project to improve and finalize the version.

Remember to keep your code clean and well-documented, use meaningful variable and function names, include comments explaining your logic, and write unit tests to ensure your code works as expected.

Happy coding!

Last Updated: 4/11/2024

---