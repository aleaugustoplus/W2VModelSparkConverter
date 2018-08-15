package com.main;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

import org.apache.spark.mllib.feature.Word2VecModel;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SparkSession;


import scala.Predef;
import scala.Tuple2;
import scala.collection.JavaConverters;


import java.util.logging.Logger;
import java.util.zip.GZIPInputStream;

/*
 * This software is meant to be a utility for converting word2vec models binary trained models 
 * to spark mllib word2vec models it is INSANE how much memory does spark take to save the 
 * binary model although it works!  
 */
public class W2VModelSparkConverter {
	
	private final static Logger log = Logger.getLogger(W2VModelSparkConverter.class.getName());
	private static final int SIZE_ARRAY = 300; 
	
	public static <A, B> scala.collection.immutable.Map <A, B> toScalaMap(Map<A, B> m) {
	    return JavaConverters.mapAsScalaMapConverter(m).asScala().toMap(
	      Predef.<Tuple2<A, B>>conforms()
	    );
	 }
	/*
	 * This requires incredible amount of memory on the dataframe creation this 
	 * is my command line I used to convert the model GoogleNews-vectors-negative300.txt
	 * spark-submit --master local[*] --num-executors 2 --driver-memory 100G \ 
	 * --executor-memory 40G W2VModelSparkConverter-0.0.1-SNAPSHOT-jar-with-dependencies.jar \
	 *  GoogleNews-vectors-negative300.txt.gz \
	 *  Output.parquet"
	 * 
	 */	
	public static void main(String []args) throws FileNotFoundException, IOException {
		
		Map<String, float[]> words = new HashMap<String, float[]>();
		log.info("Reading google text wor2vec model...");
		log.info("Input: " + args[0]);
		log.info("Output: " + args[1]);
		log.info("Starting apache spark");
        try(SparkSession spark = SparkSession
				  .builder()
				  .appName("W2VModelSparkConverter")	
				  .master("local")
				  .getOrCreate()) {
        	
        	
			try(BufferedReader file = 
					new BufferedReader(
							new InputStreamReader(
									new GZIPInputStream(
											new FileInputStream(args[0]))));
			    Scanner scan = new Scanner(file)) {
				
				int i = 0;
				scan.nextLine();
				while (scan.hasNext()) {
					String key = scan.next();
					float []array = new float[SIZE_ARRAY];
					for(int x=0; x < SIZE_ARRAY; x++) {
						try {
							array[x] = scan.nextFloat();
						} catch(Exception ex) {
							log.info("Error in: " + key + " pos: " + x + "words: " + i);
						}					
					}
					words.put(key, array);				
					i++;	
					if(i % 50000 == 0) {
						log.info("Loaded: " + i);
						System.gc();
					}
				}						
			}	
			System.gc();
		log.info("Finished google text wor2vec model!");		     
		    scala.collection.immutable.Map scalaMap = toScalaMap(words);
		    words = null;
		    System.gc();
        	Word2VecModel model = new Word2VecModel(scalaMap);
        	SQLContext sqlContext = spark.sqlContext();
        	model.save(spark.sparkContext(), args[1]);
        }        
	}
}
