# Machine Learning using Synthesized Patient Health Records on IBM DB2 🚀 🌁

---

> 📌 DISCLAIMER: This notebook is used for demonstrative and illustrative purposes only and does not constitute an offering that has gone through regulatory review. It is not intended to serve as a medical application. There is no representation as to the accuracy of the output of this application and it is presented without warranty.

This notebook explores how to train a machine learning model to predict type 2 diabetes using synthesized patient health records. The use of synthesized data allows us to learn about building a model without any concern about the privacy issues surrounding the use of real patient health records.

When the reader has completed this Code Pattern, they will understand how to:

 - Store data on IBM DB2 database
 - Prepare data using SciKit-learn and Pandas.
 - Visualize data relationships using Seaborn & Matplotlib.
 - Train a machine learning model and publish it in the Watson Machine Learning (WML) repository.
 - Deploy the model as a web service and use it to make predictions.

## Flow 
---
[![N|Solid](https://raw.githubusercontent.com/ademaldemir/machine-learning-patient-records-on-db2/master/images/flow.png?token=AFUJ44XZNS7AGBV7HXVOX2C7ITQDW)](https://nodesource.com/products/nsolid)

1. Create a Watson Studio Project on IBM Cloud.
2. Create a DB2 on Cloud Database : IBM DB2 stores information that will be used for machine learning and predictions.
3. Create Watson Machine Learning Service
2. Create a notebook in Watson Studio
3. Create a Jupyter Notebook
6. Publish and deploy model with Watson Machine Learning

## Steps
---
1. [Clone the repo](#1-clone-the-repo)
2. [Create an IBM Cloud account](#2-create-an-ibm-cloud-account)
3. [Load data into IBM Db2 on Cloud](#3-load-data-into-ibm-db2-on-cloud)
4. [Setup Watson studio project](#4-setup-watson-studio-project)
5. [Creating and deploying a machine learning model](#5-creating-and-deploying-a-machine-learning-model)
6. [Testing using UI](#5-testing-using-ui)

### 1. Clone the repo

Before we start anything, we need to clone the repo. The repo has our dataset and python notebook which we will use when creating our model.

```sh
git clone https://github.com/ademaldemir/machine-learning-patient-records-on-db2.git
```

### 2. Create an IBM Cloud account
Create a free IBM on Cloud Account if you don't already have one using the following link:

[IBM Cloud](https://cloud.ibm.com)

Creating this account will give us access to `Db2 on Cloud` and `Watson Studio` services.

### 3. Load data into IBM Db2 on Cloud

Now that we have created our IBM Cloud account. We need to create a Db2 on Cloud service. Once we have create that, we will then we able to load our data into our database.

1. [Create Db2 on Cloud Service](#3a-create-db2-on-cloud-service)
2. [Get Db2 on Cloud credentials](#3b-get-db2-on-cloud-credentials)
3. [Load Data into Db2 on Cloud](#3c-load-data-into-db2-on-cloud)
#### 3a. Create Db2 on Cloud Service

Go to the [dashboard](https://cloud.ibm.com) of your IBM Cloud account and follow the steps to create your Db2 On Cloud service.

![Searching For Db2 Service](https://raw.githubusercontent.com/ademaldemir/machine-learning-patient-records-on-db2/master/images/img-2.png)

* In the search bar at the top of your dashboard, search `Db2`.
* Although there are different database options to choose from, for the purposes of this tutorial we will be using the `Db2` option. Click `Db2` when that option appears in the search bar.

![Creating Db2 Service](https://raw.githubusercontent.com/ademaldemir/machine-learning-patient-records-on-db2/master/images/img-3.png)

* For the service name, enter in `Data-Science-for-health`.
* Make sure you pick the region that is closest to where you currently reside.
* Look at the `Pricing Plan` section and choose the `Lite` plan.
* Click `Create`

>NOTE: You will be only able to create one instance per account.

* Once you have created your database instance, we can go back to the dashboard and click on the `Resource List` link under the `Navigation Menu` section. You should then be able to see and verify that your Db2 instance has been created under the `Services` tab.  

![Check Db2 Service](https://raw.githubusercontent.com/ademaldemir/machine-learning-patient-records-on-db2/master/images/img-7.png)



#### 3b. Get Db2 on Cloud credentials

* Go to the dashboard of your IBM Cloud account and follow the steps to load your data onto Db2 On Cloud service.

* In the search bar, search `Data-Science-for-health` and click on your Db2 on Cloud service.

Before we load data to database, we need to first create credentials for our database so both that Watson Studio can connect to it and we can learn .


* Click on `Service Credentials` on the left hand side.
* Click on `New Credentials` and then `Add`. Make sure to save the credentials for later use.

![Db2 On Cloud Credentials](https://raw.githubusercontent.com/ademaldemir/machine-learning-patient-records-on-db2/master/images/img-5.png)

The key information that is important for us is:

1. `HOSTNAME`
2. `UI`
3. `PWD`
4. `DATABASE`
5. `USERNAME` ⭐️

> The important part here is, for now:
We will use the username at the bottom of the service credentials you created when uploading data to db2.
Notice ! The name of the DB2 scheme where you will install Data should be the `username` name assigned specifically for you.

![Get username](https://raw.githubusercontent.com/ademaldemir/machine-learning-patient-records-on-db2/master/images/img-6.png)

#### 3c. Load Data into Db2 on Cloud

* Click on `Open Console` button under the `Manage` on the left side and `Open Console` button will direct you to the Db2 on Cloud Console.


![Open Console](https://raw.githubusercontent.com/ademaldemir/machine-learning-patient-records-on-db2/master/images/img-4.png)

* Click on `Load` under the Hamburger menu.
* Click on `browse files` and select `diabetes.csv` from your computer.
* Click `Next`.

![Select DataSet](https://raw.githubusercontent.com/ademaldemir/machine-learning-patient-records-on-db2/master/images/img-9.png)

* The next step is to decide where our data will be stored. Click on the scheme with the same name as your DB2 username, then select `New Table`.
* Enter `DIABETES` as our table name and select `Create` and finally `Next`.
* Make sure the column names and datatypes are correct, and click `Next`.
* Click `Begin Load`.

![Load Data](https://raw.githubusercontent.com/ademaldemir/machine-learning-patient-records-on-db2/master/images/img-10.png)

Once the job has been completed, our data has finally been loaded into our database.

![Data Load Succeeded](https://raw.githubusercontent.com/ademaldemir/machine-learning-patient-records-on-db2/master/images/img-11.png)


### 4. Setup Watson Studio project

Setting up our project enivorment can be broken down in the follow steps.

1. [Creating Watson Studio service](#4a-creating-watson-studio-service)
2. [Creating a Watson Studio project](#4b-creating-a-watson-studio-project)
3. [Connect Db2 on Cloud with Watson Studio](#4c-connect-db2-on-cloud-with-watson-studio)

#### 4a. Creating Watson Studio service

* Go to Catalog and search for  `Watson Studio` and click on that option.

![Creating Watson Studio Service](https://raw.githubusercontent.com/ademaldemir/machine-learning-patient-records-on-db2/master/images/img-12.png)

* Fill out all the fields and choose 'Lite' plan.
* Click `Create` and then `Get Started`.

This will redirect you to the Watson Studio homepage.

#### 4b. Creating a Watson Studio project

Hey! 🧞‍♂️
Welcome to the new interface of IBM Cloud Pak for Data. ❤️

![IBM-Cloud-Pak-For-Data](https://raw.githubusercontent.com/ademaldemir/machine-learning-patient-records-on-db2/master/images/img-13.png)

Let's now create and setup our project.

![Creating Project](https://raw.githubusercontent.com/ademaldemir/machine-learning-patient-records-on-db2/master/images/img-14.png)

* Select `Create a Project` and then select `Standard`.
* If you haven't created object storage earlier, go to the bottom of the page and click the link `Cloud Object Storage`. Choose the `Lite` plan and click `Create`.
* Go back to the project page and make sure to choose the Cloud Object Storage that you have created earlier.
* Fill out the project details and click `Create`.

This will take you to your project dashboard/homepage.

#### 4c. Connect Db2 on Cloud with Watson Studio

* On the top of the project homepage, select `Add to project` and then click `Connection`.
* Select  `Db2` connection option that you created.

![Connection Database](https://raw.githubusercontent.com/ademaldemir/machine-learning-patient-records-on-db2/master/images/img-15.png)

* This will take you to a connection configuration page. Here, we will enter the Db2 credentials that we got from Step 4a. Make sure to use `50000` or it can be default value for the `Port` option.
* Click `Create` once you have entered all the required information.

This will redirect you to the asset page for this project, and you should see your new Db2 connection as one of the assets.

Now that we have our database connected to our project, we need to also connect our data that is stored in our database to the project as well.

* On the top of the project homepage, select `Add to project` and then click `Connected data`.
* Select `Select Source`.

![Connecting Data](https://raw.githubusercontent.com/ademaldemir/machine-learning-patient-records-on-db2/master/images/img-16.png)

* Select our database, scheme and finally our table `DIABETES`.
* Click `Select`.
* Let's name this connected data as `DIABETES` and then click `Create`.

We have finally created our Watson Studio service. Within that, created a project where our database and data are connected. We can now finally start coding and building our Machine Learning models on Jupyter Notebook!

### 5. Creating and deploying a Machine Learning Model

* On the top of the project homepage, select `Add to project` and then click `Notebook`.

![Creating Notebook](https://raw.githubusercontent.com/ademaldemir/machine-learning-patient-records-on-db2/master/images/img-17.png)

* Fill out the notebook details
* Select `From URL` option from the tab and paste the following link to Notebook URL field:
`https://raw.githubusercontent.com/ademaldemir/machine-learning-patient-records-on-db2/master/health_record_notebook.ipynb`

Before we run the notebook, we need to create `Watson Machine Learning` instance so that we can deploy the model to Watson Machine Learning on IBM Cloud. Here are the steps:

* Go to IBM cloud dashboard and click `Create Resource`
* Search for `machine learning` and select `Machine Learning` service
* Fill out the details, select `Lite` plan and click `Create`.

![Create WML service](https://raw.githubusercontent.com/ademaldemir/machine-learning-patient-records-on-db2/master/images/img-18.png)

* And finally Create `Service Credentials` as shown below

![Create Credentials](https://raw.githubusercontent.com/ademaldemir/machine-learning-patient-records-on-db2/master/images/img-19.png)

In the notebook, after the `import` cell, add cell to create connection as shown below.

![Add Db2 Connections](https://raw.githubusercontent.com/ademaldemir/machine-learning-patient-records-on-db2/master/images/img-20.png)

Select your DB2 `username` from the Schemes section. And select the table named `DIABETES` you loaded.

![Select-table](https://raw.githubusercontent.com/ademaldemir/machine-learning-patient-records-on-db2/master/images/img-21.png)

Run the cell containing the functions imported from the `ibmdbpy` library and your db2 informations. And observe the results. 🔎

![observe-data](https://raw.githubusercontent.com/ademaldemir/machine-learning-patient-records-on-db2/master/images/img-22.png)

The dataset type is `ibmdbpy.frame.IdaDataFrame`.

![dataset-type](https://raw.githubusercontent.com/ademaldemir/machine-learning-patient-records-on-db2/master/images/img-23.png)

To make the dataset type `pandas.core.frame.DataFrame`, we click on the **Find and add data** button in the upper right corner and select **pandas DataFrame** from the *Connections* tab. Then, select your db2 username assigned specifically to you from the schemas. Find the **DIABETES** table and click Select.

![turn-dataset-type-to-pandas](https://raw.githubusercontent.com/ademaldemir/machine-learning-patient-records-on-db2/master/images/img-24.png)


Add Watson Machine Learning service credentials that you have saved from above, at step 5 of the notebook as shown below.

![Add WML Creds](https://raw.githubusercontent.com/ademaldemir/machine-learning-patient-records-on-db2/master/images/img-25.png)

Then, run all the cells. At the end of the run the model will be deployed using Watson Machine Learning on IBM Cloud so that you could use the same model to predict the person's illness through Watson Machine Learning service.

## Learn more
---

* Follow the IBM Z & LinuxONE Community Turkey community.
* Check out our other [events.](https://www.meetup.com/IBM-Z-LinuxONE-Community-Turkey)

---
🚀 created by: [Adem Aldemir](https://www.linkedin.com/in/ademaldemir/) 👨🏻‍💻

---

![IBM](https://raw.githubusercontent.com/ademaldemir/machine-learning-patient-records-on-db2/master/images/iloveai.jpg)


