# I3_MLOps
## Setup
In order to run the testOfTool_V2, users will need to set environmental variables to connect to database. 

## testOfTool
This file connects to the database and runs the query to get user information and requests made to the service. After running the query, we do some data processing and introduce some bias for the sake of this exercise and example. 

## runTests
This file is where we utilize AIF360. We use the csv saved from running testOfTool and utilize it to find, mitigate, and validate mitigation of bias. 
