package software.nps.visionlab.sat_example.bin

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StringType
import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.fs.Path
import org.apache.hadoop.mapred.JobConf
import org.apache.hadoop.io.ArrayWritable
import org.apache.hadoop.io.BytesWritable
import org.apache.hadoop.io.NullWritable
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.Job

import software.uncharted.salt.core.projection.numeric._
import software.uncharted.salt.core.generation.request._
import software.uncharted.salt.core.generation.Series
import software.uncharted.salt.core.generation.TileGenerator
import software.uncharted.salt.core.generation.output.SeriesData
import software.uncharted.salt.core.analytic.numeric._

import java.io._
import java.awt.geom.Point2D

import scala.util.parsing.json.JSONObject
import org.gdal.gdal.Dataset


object Main {

  // Defines the tile size in both x and y bin dimensions
  val tileSize = 256

  // Defines the output layer name
  val layerName = "satData"

  def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
      val p = new java.io.PrintWriter(f)
      try {
          op(p)
      } finally {
          p.close()
      }
  }

  // Creates and returns an Array of Double values encoded as 64bit Integers
  def createByteBuffer(tile: SeriesData[(Int, Int, Int), (Int, Int), Double, (Double, Double)]): Array[Byte] = {
    val byteArray = new Array[Byte](tileSize * tileSize * 8)
    var j = 0
    tile.bins.foreach(b => {
      val data = java.lang.Double.doubleToLongBits(b)
      for (i <- 0 to 7) {
        byteArray(j) = ((data >> (i * 8)) & 0xff).asInstanceOf[Byte]
        j += 1
      }
    })
    byteArray
  }


  def main(args: Array[String]): Unit = {
    if (args.length < 2) {
      println("Requires commandline: <spark-submit command> inputFilePath outputPath")
      System.exit(-1)
    }

    val inputPathName = args(0)
    val outputPath = args(1)

    val conf = new SparkConf().setAppName("salt-bin-example")
    conf.set("spark.kryoserializer.buffer.max", "1000MB")
    conf.set("spark.executor.memory", "7GB")
    conf.set("spark.executor.extraLibraryPath", "/home/trbatcha/tools/lib")
    conf.set("spark.executor.extrajavaoptions", "-XX:+UseConcMarkSweepGC")
    conf.set("log4j.logger.org.apache.spark.rpc.akka.ErrorMonitor", "FATAL")
    val sc = new SparkContext(conf)
    val hadoopConf = sc.hadoopConfiguration
    //val hjob = new Job(hadoopConf);
    val hjob = Job.getInstance(hadoopConf)
    val fs = org.apache.hadoop.fs.FileSystem.get(hadoopConf)
    val inputPath = new Path(inputPathName)
    val exists = fs.exists(inputPath)
    if (!exists){
        println( inputPathName + " does not exist!!")
        System.exit(1)
    }else {
        println("Processing " + inputPathName)
    }
    FileInputFormat.setInputPaths(hjob, inputPath)
    val files = fs.listFiles(inputPath, true)
    while (files.hasNext()) {
        val next = files.next()
        println("next " + next.getPath().getName())
        if (next.isFile()){
            val name = next.getPath().getName()
            val ext  = name.split(',').drop(1).lastOption
            if (ext == "ntf" || ext == "tif") {
                FileInputFormat.addInputPath(hjob, next.getPath())
            }
        }
    }
    val resData = sc.newAPIHadoopRDD(hjob.getConfiguration(),
                  classOf[WholeFileInputFormat],
                  classOf[String],classOf[ArrayWritable]).map(n => {
                    val fname = n._1
                    println(fname)
                    var data = n._2
                    var bytes = data.get().asInstanceOf[Array[BytesWritable]]
                    println("array bytes length " + bytes.length)
                    var out = None: Option[FileOutputStream]
                    val tname = "/dev/shm/" + fname
                    var bcnt = 0
                    try {
                        out = Some(new FileOutputStream(tname))
                        for (i <- 0 until bytes.length) {
                            val next = bytes(i).copyBytes()
                            bcnt += next.length
                            out.get.write(next)
                        }
                    } catch {
                        case e: IOException => e.printStackTrace
                    } finally {
                        if (out.isDefined)
                            out.get.close
                    }
                    println("wrote temp file of size " + bcnt)
                    var nfile = new File(tname)
                    var points : Array[java.awt.geom.Point2D] = 
                        Array(new java.awt.geom.Point2D.Double(0.0, 0.0), 
                              new java.awt.geom.Point2D.Double(0.0, 0.0), 
                              new java.awt.geom.Point2D.Double(0.0, 0.0), 
                              new java.awt.geom.Point2D.Double(0.0, 0.0)) 
                    var props = None : Option[java.util.Properties]
                    try {
                        var gfoot = new GDALFootprint()
                        val npoints = gfoot.getCorners(nfile)
                        if (npoints != null)
                            points = npoints
                        props = Some(gfoot.getLastProps())
                    } catch {
                        case e: IllegalArgumentException => {}
                    } finally {
                        nfile.delete()
                    }
                    println(fname)
                    Row(fname, String.valueOf(points(0).getX()), 
                               String.valueOf(points(0).getY()),
                               String.valueOf(points(1).getX()), 
                               String.valueOf(points(1).getY()),
                               String.valueOf(points(2).getX()), 
                               String.valueOf(points(2).getY()),
                               String.valueOf(points(3).getX()), 
                               String.valueOf(points(3).getY()))
                      
                  }).persist()

    val schemaString = "name x0x x0y x1x x1y x2x x2y x3x x3y"
    val schema = StructType(
        schemaString.split(" ").map(fieldName => StructField(fieldName,
                                                            StringType, true)))

    val sqlContext = new SQLContext(sc)
    val dFrame = sqlContext.createDataFrame(resData, schema)
    dFrame.registerTempTable("corners")


    //val nameRDD = sc.binaryFiles("myntf/blk_3")

    /*
    sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(s"file://$inputPath")
      .registerTempTable("taxi_micro")
    */

    // Construct an RDD of Rows containing only the fields we need. Cache the result
    val input = sqlContext.sql("select cast(x0x as double), cast(x0y as double), cast(x1x as double), cast(x1y as double), cast(x2x as double), cast(x2y as double), cast(x3x as double), cast(x3y as double) from corners")
      .rdd.cache()

    // Given an input row, return pickup longitude, latitude as a tuple
    val pickupExtractor = (r: Row) => {
      if (r.isNullAt(0) || r.isNullAt(1) || r.isNullAt(2) || r.isNullAt(3) ||
          r.isNullAt(4) || r.isNullAt(5) || r.isNullAt(6) || r.isNullAt(7)
         ) {
        None
      } else {
        Some((r.getDouble(0), r.getDouble(1), r.getDouble(2), r.getDouble(3),
              r.getDouble(4), r.getDouble(5), r.getDouble(6), r.getDouble(7)))
      }
    }

    // Tile Generator object, which houses the generation logic
    val gen = TileGenerator(sc)

    // Break levels into batches. Process several higher levels at once because the
    // number of tile outputs is quite low. Lower levels done individually due to high tile counts.
    val levelBatches = List(List(0, 1, 2, 3, 4, 5, 6, 7, 8), List(9, 10, 11), List(12), List(13))

    // Iterate over sets of levels to generate.
    val levelMeta = levelBatches.map(level => {

      println("------------------------------")
      println(s"Generating level $level")
      println("------------------------------")

      // Construct the definition of the tiling jobs: pickups
      val pickups = new Series((tileSize - 1, tileSize - 1),
        pickupExtractor,
        new MercatorRectProjection(1024, level),
        (r: Row) => Some(1),
        CountAggregator,
        Some(MinMaxAggregator))

      // Create a request for all tiles on these levels, generate
      val request = new TileLevelRequest(level, (coord: (Int, Int, Int)) => coord._1)
      val rdd = gen.generate(input, pickups, request)

      // Translate RDD of Tiles to RDD of (coordinate,byte array), collect to master for serialization
      val output = rdd
        .map(s => pickups(s).get)
        .map(tile => {
          // Return tuples of tile coordinate, byte array
          (tile.coords, createByteBuffer(tile))
        })
        .collect()

      // Save byte files to local filesystem
      output.foreach(tile => {
        val coord = tile._1
        val byteArray = tile._2
        val limit = (1 << coord._1) - 1
        // Use standard TMS path structure and file naming
        println("saving to file " + s"$outputPath/$layerName")
        val file = new File(s"$outputPath/$layerName/${coord._1}/${coord._2}/${limit - coord._3}.bins")
        file.getParentFile.mkdirs()
        val output = new FileOutputStream(file)
        output.write(byteArray)
        output.close()
      })

      // Create map from each level to min / max values.
      rdd
        .map(s => pickups(s).get)
        .map(t => (t.coords._1.toString, t.tileMeta.get))
        .reduceByKey((l, r) => {
          (Math.min(l._1, r._1), Math.max(l._2, r._2))
        })
        .mapValues(minMax => {
          JSONObject(Map(
            "min" -> minMax._1,
            "max" -> minMax._2
          ))
        })
        .collect()
        .toMap
    })

    // Flatten array of maps into a single map
    val levelInfoJSON = JSONObject(levelMeta.reduce(_ ++ _)).toString()
    // Save level metadata to filesystem
    val pw = new PrintWriter(s"$outputPath/$layerName/meta.json")
    pw.write(levelInfoJSON)
    pw.close()

    sc.stop()

  }
}
