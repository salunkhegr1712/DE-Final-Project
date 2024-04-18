package org.example;

import org.apache.spark.SparkConf;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.expressions.WindowSpec;

import static org.apache.spark.sql.functions.*;

// Press Shift twice to open the Search Everywhere dialog and type `show whitespaces`,
// then press Enter. You can now see whitespace characters in your code.
public class Main {
    public static <JavaRDD> void main(String[] args) {

        SparkConf conf = new SparkConf().setAppName("StockMarketCryptoAnalysi").setMaster("local");
        SparkSession spark = SparkSession.builder().config(conf).getOrCreate();

        // Read Apple stock data
        Dataset<Row> appleDF = spark.read().option("header", "true")
                .option("inferSchema","true")
                .csv("MyResources/aapl_2014_2023.csv");
//        appleDF.show();

        //Read bitcon crypto data
        Dataset<Row> bitconDF=readCryptoDataset(spark,"MyResources/Bitcoin Historical Data2.csv");
        Dataset<Row> ethereumDF = readCryptoDataset(spark,"MyResources/Ethereum Historical Data2.csv");

        // Calculate 20-day moving average per year for Apple stock
        Dataset<Row> appleMovingAvgDF = calculateMovingAverage(appleDF, "apple");

//        appleMovingAvgDF.show(20);
        // Calculate 20-day moving average per year for Bitcoin
        Dataset<Row> bitcoinMovingAvgDF = calculateMovingAverage(bitconDF, "bitcoin");

        // Calculate 20-day moving average per year for Ethereum
        Dataset<Row> ethereumMovingAvgDF = calculateMovingAverage(ethereumDF, "ethereum");

        // Join the moving averages with the original data
        Dataset<Row> resultDF = joinMovingAverages(appleMovingAvgDF, bitcoinMovingAvgDF, ethereumMovingAvgDF);

//        resultDF.show();
        // Write the result to a CSV file
        resultDF.repartition(1).write().csv("outputs/final-project-output.csv");


        // Displaying the first few rows of the result
        resultDF.show();
    }
    private static Dataset<Row> readCryptoDataset(SparkSession spark,String filePath){
        return spark.read().option("header", "true")
                .csv(filePath)
                .withColumn("Date", to_date(col("Date"), "MM/dd/yyyy"))
                .withColumn("year", year(col("Date")))
                .withColumn("Price", regexp_replace(col("Price"), ",", "").cast("double"))
                .withColumn("Open", regexp_replace(col("Open"), ",", "").cast("double"))
                .withColumn("High", regexp_replace(col("High"), ",", "").cast("double"))
                .withColumn("Low", regexp_replace(col("Low"), ",", "").cast("double"))
                .withColumn("Volume",when(col("`Vol.`").contains("B"),regexp_replace(col("`Vol.`"), "B", "").cast("double")
                        .multiply(lit(1000000000))).otherwise(when(col("`Vol.`").contains("M"),regexp_replace(col("`Vol.`"), "M", "").cast("double")
                        .multiply(lit(1000000))).otherwise(regexp_replace(col("`Vol.`"), "K", "").cast("double")
                        .multiply(lit(1000)))))
                .withColumn("Change %", regexp_replace(col("Change %"), "%", "").cast("double"))
                .select(col("Date").as("date"), col("Price").as("price"), col("Open").as("open"),
                        col("High").as("high"), col("Low").as("low"), col("Volume").as("volume"),
                        col("Change %").as("change_percent"), col("year"));
    }

    private static Dataset<Row> calculateMovingAverage(Dataset<Row> df, String prefix) {
        df = df.withColumn("year", year(col("Date")));
        WindowSpec windowSpec = Window.partitionBy("year").orderBy("date").rowsBetween(-19, 0);
        return df.withColumn(prefix + "_20day_avg", avg(col("high")).over(windowSpec));
    }

    private static Dataset<Row> joinMovingAverages(Dataset<Row> appleMovingAvgDF, Dataset<Row> bitcoinMovingAvgDF, Dataset<Row> ethereumMovingAvgDF) {
        // Ensure that the column names are correctly specified
        appleMovingAvgDF = appleMovingAvgDF
                .withColumnRenamed("high", "apple_high")
                .withColumnRenamed("low", "apple_low")
                .drop("year");

        bitcoinMovingAvgDF = bitcoinMovingAvgDF
                .withColumnRenamed("high", "bitcoin_high")
                .withColumnRenamed("low", "bitcoin_low")
                .drop("year");

        ethereumMovingAvgDF = ethereumMovingAvgDF
                .withColumnRenamed("high", "ethereum_high")
                .withColumnRenamed("low", "ethereum_low");

        // Join the three DataFrames on the 'date' and 'year' columns
        return appleMovingAvgDF
                .join(bitcoinMovingAvgDF, "date")
                .join(ethereumMovingAvgDF, "date")
                .select("date", "year", "apple_high", "apple_20day_avg", "apple_low",
                        "bitcoin_high", "bitcoin_20day_avg", "bitcoin_low",
                        "ethereum_high", "ethereum_20day_avg", "ethereum_low");
    }
}