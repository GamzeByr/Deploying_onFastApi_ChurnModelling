## Creating a Machine Learning Model (basic) and deploying on FastApi (local)
## Using Mlflow to save experiences
## Using MySql database to determine drifts (basic) 


##1. Start docker-compose
```
docker-compose up -d

##2.Create database, user

# Connect mysql container 
docker exec -it mysql -u root -p

# Create database 
mysql> create database database_name;

# Show databases
mysql> show databases;

# Create user 
mysql> CREATE USER 'user_name'@'%' IDENTIFIED BY 'password';

# See users
mysql> select user, host from mysql.user;

# Grand user on database
mysql> GRANT ALL PRIVILEGES ON database_name.* TO 'user_name'@'%' WITH GRANT OPTION;

mysql> FLUSH PRIVILEGES;

mysql> exit
```

## 3. Develop the App

## 4. Copy/Push App to VM 

## 5. Install requirements

## 6. Run uvicorn 
```commandline
uvicorn main:app \
--host 0.0.0.0 \
--port 8002 \
--reload
```
## 7. Check customers table in mysql 
```
$ docker exec -it mysql -u user_name -D database_name -p
Enter password: password

mysql> use database_name;
Database changed

mysql> show tables;
+-------------------------+
| Tables_in_database_name |
+-------------------------+
| customers               |
+-------------------------+
1 row in set (0.00 sec)

mysql> select * from customers;
Empty set (0.00 sec)

mysql> describe customers;
+------------------+--------------+------+-----+---------+----------------+
| Field            | Type         | Null | Key | Default | Extra          |
+------------------+--------------+------+-----+---------+----------------+
| customerId       | int          | NO   | PRI | NULL    | auto_increment |
| customerFName    | varchar(50)  | YES  |     | NULL    |                |
| customerLName    | varchar(50)  | YES  |     | NULL    |                |
| customerEmail    | varchar(50)  | YES  |     | NULL    |                |
| customerPassword | varchar(255) | YES  |     | NULL    |                |
| customerStreet   | varchar(100) | YES  |     | NULL    |                |
| customerCity     | varchar(20)  | YES  |     | NULL    |                |
| customerState    | varchar(5)   | YES  |     | NULL    |                |
| customerZipcode  | varchar(10)  | YES  |     | NULL    |                |
+------------------+--------------+------+-----+---------+----------------+
9 rows in set (0.01 sec)

mysql> exit
```
## 10. Check port forwarding

## 11. Open browser
http://localhost:8002/docs






