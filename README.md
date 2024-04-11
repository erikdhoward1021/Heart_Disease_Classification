# How to Navigate the Standard Project Template & General Guidelines

The structure of this project follows a standard structure to ensure consistency and ease of navigation. Here's what should be saved in each folder:

## `/src` (Source Code)
This folder should contain all the source code files for your project. This includes all the scripts, modules, and packages that you write.

## `/tests` (Unit Tests)
This folder should contain all the unit tests for your project. Each module in the `/src` directory should have a corresponding test file in this directory.

## `/docs` (Documentation)
This folder should contain all the documentation for your project. This includes user guides, API documentation, and any other information that helps others understand and use your project.

## `/data/raw` (Raw Data)
This subfolder should contain the original, unmodified data files. This could include CSV files, databases, or any other data sources. If necessary, please do not include very large data files in your Git repository. Instead, provide instructions on how to download or generate the data.

## `/data/processed` (Processed Data)
This subfolder should contain the data that has been cleaned, preprocessed, or otherwise transformed for analysis or model training. This ensures a clear distinction between the original data and the data that you actually use in analyses or models. If necessary, please do not include very large data files in your Git repository. Instead, provide instructions on how to download or generate the data.

## `/notebooks` (Jupyter Notebooks)
This folder should contain all Jupyter notebooks used for exploratory data analysis, model development, and other research. Each notebook should be named clearly and include comments explaining what each section of code does.

## `/models` (Model Files)
This folder should contain the saved model files. This could include trained machine learning models, model architectures, and other related files.

## `/scripts` (Utility Scripts)
This folder should contain any utility scripts that are used for tasks like data preprocessing, model training, etc. These scripts should be well-documented and reusable.

## `/results` (Results and Outputs)
This folder should contain the output of your data analysis and model training, such as figures, tables, performance metrics, etc. Do not include large output files in your Git repository. Instead, provide instructions on how to generate these outputs.

Remember to always keep your code clean and well-documented. Use meaningful variable and function names, include comments explaining your logic, and write unit tests to ensure your code works as expected.


# To use this Template:
1. Create a new repository on GitHub. When creating a new repository on GitHub, there's an option to create a repository from a template. Select this repository.

2. Clone the new repository locally. You can use the git clone command to clone the new repository to their local machine.

3. Customize the template. You can now customize the template to fit your specific project needs. This could involve adding new files, modifying existing ones, or deleting unnecessary ones.

# Here is an example layout

## [Project Name]
The official repository for [project name] by Government Solutions &amp; Innovations - DSIT.

## Everything in this repository & project assumes:
- Clear understanding of intended usage (how will they actually use it?)
- Strong & captured success criteria (what does success look like?)
- Agreed upon & ballpark estimates for ROI KPI (how will we actually measure success?)


## Step-wise, let's aim for the following goals:
1. Complete pipeline (though poorly trained):
   - This is not to train the best model, rather to pull the thread fully through to:
      - understand what does/doesn't work
      - what is easy/difficult to address
      - develop the general framework of each major portion
2. Hands-off-keyboard evaluation to determine if:
   - this is a viable project
   - we need more data
   - we need to be creative with feature engineering
   - etc.
3. Regrouping with end-user group to explain current status & anticipated path forward
4. Hands-on work to improve and finalize v1


## Please use branching when developing new features or fixing bugs:
1. Create new branch (either done above [click on 'main' to add a new branch] or, in the terminal, with 'git checkout -b <new_branch_name>)
#### To swap to your feature branch in AWS, use the terminal to:
2. cd into Repository
3. git checkout -b <branch_name>
#### To save your work locally (in your AWS local repository):
4. git add <file_you_are_working_on>
5. git commit -m 'You commit message here'
6. git status (this will confirm that your commit did in fact make changes to your local repository)
#### After you have completed the feature / bug fix, push your changes to YOUR branch on 'origin' (i.e., here in GitHub):
6. git push origin <branch_name>

This will merge the changes into YOUR branch here in GitHub. In order to merge those changes into 'main', submit a pull request and add a reviewer.

Big picture, we will only merge changes to main once we have reviewed, tested, and confirmed the updates you've made in your branch.

Last Updated 4/8/2024
