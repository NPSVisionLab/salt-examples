package software.nps.visionlab.sat_example.bin

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.SparkFiles
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StringType
import org.apache.spark.storage.StorageLevel
import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.fs.Path
import org.apache.hadoop.mapred.JobConf
import org.apache.hadoop.io.ArrayWritable
import org.apache.hadoop.io.BytesWritable
import org.apache.hadoop.io.NullWritable
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat
import org.apache.hadoop.mapreduce.Job
import org.apache.hadoop.conf.Configuration
import org.apache.commons.io.FileUtils
import org.apache.commons.io.filefilter.SuffixFileFilter
import org.apache.commons.io.filefilter.WildcardFileFilter

import software.uncharted.salt.core.projection.numeric._
import software.uncharted.salt.core.generation.request._
import software.uncharted.salt.core.generation.Series
import software.uncharted.salt.core.generation.TileGenerator
import software.uncharted.salt.core.generation.output.SeriesData
import software.uncharted.salt.core.analytic.numeric._

import java.io._
import java.awt.geom.Point2D
import java.awt.geom.Rectangle2D
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.attribute.PosixFilePermission
import java.nio.file.attribute.PosixFilePermissions
import java.util.Set

import scala.util.parsing.json.JSONObject
import scala.collection.JavaConversions._
import scala.collection.mutable.ArrayBuffer
import org.gdal.gdal.Dataset



class MultiWriter(suffix: String) {
    private val writers = scala.collection.mutable.Map[String, java.io.PrintWriter]()
    def write(key: String, value: String) {
        if (!writers.contains(key)) {
            val f = new java.io.File("output/" + key + "/" +suffix)
            f.getParentFile.mkdirs
            writers(key) = new java.io.PrintWriter(f)
        }
        writers(key).println(value)
    }
    def close = writers.values.foreach(_.close)
}

object Main {

  // Defines the tile size in both x and y bin dimensions
  val tileSize = 256

  // Defines the output layer name
  val layerName = "satData"
  val imageLayerName = "iData"

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
    var detectOnly = false

    if (args.length == 3) {
        if (args(2) == "detectOnly") {
            detectOnly = true
        }
    }

    println("Removing old tiles " + outputPath + "/" + layerName)
    var layerPath = new File(outputPath + "/" + layerName)
    var imageLayerPath = new File(outputPath + "/" + imageLayerName)
    FileUtils.deleteDirectory(layerPath)
    FileUtils.deleteDirectory(imageLayerPath)
    layerPath.mkdirs()
    imageLayerPath.mkdirs()
    val perms = PosixFilePermissions.fromString("rwxrwxrwx");
    Files.setPosixFilePermissions(Paths.get(imageLayerPath.getAbsolutePath()), 
                                   perms)


    val conf = new SparkConf().setAppName("sat-example")
    conf.set("spark.kryoserializer.buffer.max", "1000MB")
    conf.set("spark.executor.memory", "7GB")
    // This causes the python detect code to get interrupted!
    //conf.set("spark.dynamicAllocation.enabled", "false")
    conf.set("spark.dynamicAllocation.executorIdleTimeout", "36000000")
    conf.set("spark.worker.timeout", "36000000")
    conf.set("spark.driver.maxResultSize", "1GB")
    conf.set("spark.executor.extraLibraryPath", "/home/trbatcha/tools/lib")
    conf.set("spark.executor.extrajavaoptions", "-XX:+UseConcMarkSweepGC")
    conf.set("log4j.logger.org.apache.spark.HeartbeatReceiver", "TRACE")
    //conf.set("log4j.logger.org.apache.spark.rpc.akka.ErrorMonitor", "FATAL")
    conf.set("log4j.logger.org.apache.spark.rpc.akka.ErrorMonitor", "TRACE")
    val sc = new SparkContext(conf)
    // Uncomment for more debug output on main log
    sc.setLogLevel("DEBUG")
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

    // Don't add the path since we are adding individual files
    //FileInputFormat.setInputPaths(hjob, inputPath)
    var remoteIter = fs.listFiles(inputPath, true)
    var max_count = -1
    var done = false
    while (remoteIter.hasNext() && done == false) {
        if (count > max_count && max_count > 0)
            done = true
        else {
            var f = remoteIter.next()
            println("next " + f.getPath().toString())
            if (f.isFile()){
                val name = f.getPath().getName()
                var ext = name.split('.').drop(1).lastOption
                ext = ext.map(_.toLowerCase)
                if (ext == Some("ntf") || ext == Some("tif")) {
                //if (ext == Some("ntf")) {
                    // for now only fetch pan images
                    //if (name.matches("(.*)pan(.*)") == false) {
                    count += 1
                    println("adding file " + name)
                    FileInputFormat.addInputPath(hjob, f.getPath())
                    //}
                }
            }
        }
    }
    //val detectorScript = "/home/trbatcha/salt-examples/sat-example/doDetect.py"
    val detectorScript = "/home/trbatcha/salt-examples/sat-example/doDetectNoTile.py"
    sc.addFile(detectorScript)
    val tileGenScript = "/home/trbatcha/salt-examples/sat-example/doTile.py"
    sc.addFile(tileGenScript)
    val tileScript = "/home/trbatcha/salt-examples/sat-example/gdal2tiles.py"
    sc.addFile(tileScript)
    val modelfile = "/home/trbatcha/salt-examples/sat-example/mymodel.caffemodel"
    sc.addFile(modelfile)
    val protofile = "/home/trbatcha/salt-examples/sat-example/test_ships.prototxt"
    sc.addFile(protofile)
    val cfgfile = "/home/trbatcha/salt-examples/sat-example/faster_rcnn_end2end_ships.yml"
    sc.addFile(cfgfile)

    //Build a schema for our output
    val schemaString = "name type x0x x0y x1x x1y x2x x2y x3x x3y x y w h"
    val schema = StructType(
        schemaString.split(" ").map(fieldName => StructField(fieldName,
                                                            StringType, true)))
    val sqlContext = new SQLContext(sc)

    //Add all the python files from py-faster-rcnn
    //FileUtils.listFiles(new File("/home/trbatcha/salt-examples/sat-example/lib"), 
    //               Array("py"),true).foreach( f =>
    //              {
    //                 val fullPath = f.getPath()
    //                val newPath = fullPath.replace(
    //                  "/home/trbatcha/salt-examples/sat-example/", "")
    //              println("copying " + newPath)
    //             sc.addFile(newPath)
    //        })

    val resData = sc.newAPIHadoopRDD(hjob.getConfiguration(),
                  classOf[WholeFileInputFormat],
                  classOf[String],classOf[BytesWritable]).flatMap(n => {
                    val fname = n._1
                    println(fname)
                    var data = n._2 // Data is not currently being used
                    var mres : ArrayBuffer[Row] = ArrayBuffer()
                    var nfile = new File(fname)

                    var haveCorner = false
                    var points : Array[java.awt.geom.Point2D] = 
                            Array(new java.awt.geom.Point2D.Double(0.0, 0.0), 
                                  new java.awt.geom.Point2D.Double(0.0, 0.0), 
                                  new java.awt.geom.Point2D.Double(0.0, 0.0), 
                                  new java.awt.geom.Point2D.Double(0.0, 0.0)) 
                    var props = None : Option[java.util.Properties]
                    try {
                        var gfoot = new  GDALFootprint()
                        val npoints = gfoot.getCorners(nfile)
                        if (npoints != null) {
                            var skip = false
                            // If our image wraps around the globe the
                            // don't process it since our tiling (salt)
                            // does not support it and it could hang
                            if (npoints(0).getX() > 0) {
                                if (npoints(2).getX() < 0) {
                                    skip = true
                                }
                            }
                            if (npoints(0).getX() < 0) {
                                if (npoints(2).getX()> 0) {
                                    skip = true
                                }
                            }
                            if (npoints(0).getY() > 0) {
                                if (npoints(2).getY() < 0) {
                                    skip = true
                                }
                            }
                            if (npoints(0).getY() < 0) {
                                if (npoints(2).getY() > 0) {
                                    skip = true
                                }
                            }
                            if (skip == false) {
                                haveCorner = true
                                points = npoints
                            }else {
                                println("Image: " + fname + " wraps around globe and this is not supported")
                            }
                        }
                        props = Some(gfoot.getLastProps())
                        val dscript = SparkFiles.get(detectorScript)
                        val tscript = SparkFiles.get(tileScript)
                        val mfile = SparkFiles.get(modelfile)
                        val pfile = SparkFiles.get(protofile)
                        val cfg = SparkFiles.get(cfgfile)
                        val dir = SparkFiles.getRootDirectory()
                        if (haveCorner) {
                            val detects: java.util.ArrayList
                             [GDALFootprint.DetectionResult] = 
                                    //gfoot.getDetections(dir + "/doDetect.py", 
                                    gfoot.getDetections(dir + "/doDetectNoTile.py", 
                                         dir + "/mymodel.caffemodel",
                                         dir + "/test_ships.prototxt",
                                         dir + "/faster_rcnn_end2end_ships.yml",
                                         imageLayerPath)
                            println("detects size " + String.valueOf(detects.size));
                            for (i <- 0 until detects.size) {
                                mres +=  Row(fname, detects.get(i).objName, 
                                    String.valueOf(detects.get(i).p0.getX()),
                                    String.valueOf(detects.get(i).p0.getY()),
                                    String.valueOf(detects.get(i).p1.getX()),
                                    String.valueOf(detects.get(i).p1.getY()),
                                    String.valueOf(detects.get(i).p2.getX()),
                                    String.valueOf(detects.get(i).p2.getY()),
                                    String.valueOf(detects.get(i).p3.getX()),
                                    String.valueOf(detects.get(i).p3.getY()),
                                    String.valueOf(detects.get(i).pix.getX()),
                                    String.valueOf(detects.get(i).pix.getY()),
                                    String.valueOf(detects.get(i).width),
                                    String.valueOf(detects.get(i).height)
                                    )
                            }
                        }
                        gfoot.CloseDataset()
                    } catch {
                        case e: IllegalArgumentException => {}
                    } finally {
                        //Need to keep file until after doTile
                        //Then delete it
                        //nfile.delete()
                    }
                    println(fname)
                    //println(mres(0)) 
                    if (haveCorner)
                        mres += Row(fname, "corner", 
                               String.valueOf(points(0).getX()), 
                               String.valueOf(points(0).getY()),
                               String.valueOf(points(1).getX()), 
                               String.valueOf(points(1).getY()),
                               String.valueOf(points(2).getX()), 
                               String.valueOf(points(2).getY()),
                               String.valueOf(points(3).getX()), 
                               String.valueOf(points(3).getY())
                               )
                    //println(mres(1)) 
                   
                    println("size of res " + String.valueOf(mres.length))
                    if (mres.length == 1) {
                        println("Returning only one item")
                        println(mres(0))
                    }
                    mres
                }).persist()
    if (detectOnly) {
        val outpath = new Path("output")
        if (fs.exists(outpath)) {
            fs.delete(outpath, true)
        }
        resData.saveAsTextFile("output")
    }else {

        println("Generating detection tiles")
        val genpath = new Path("genOutput")
        if (fs.exists(genpath)) {
            fs.delete(genpath, true)
        }

        resData.map(r => {
            val genscript = SparkFiles.get(tileGenScript)
            val dir = SparkFiles.getRootDirectory()
            // Only send the pixel coords and not lat/longs
            val rtype = r.getString(1)
            println("row type " + rtype)
            if (rtype == "ship"  || rtype == "fast_ship" ) {
                println("Processing a detection for tiles")
                println(r)
                val detect = r.getString(0) + " " + // file
                         r.getString(1) + " " +     // type
                         r.getString(10) + " " +    // pix x
                         r.getString(11) + " " +    // pix y
                         r.getString(12) + " " +    // width
                         r.getString(13)            // height
                println("Detect: " + detect)
                var gfoot = new  GDALFootprint()
                println("Calling doTile")
                gfoot.getTiles(dir + "/doTile.py", detect, imageLayerPath)
                detect
            } else {
                "corner"
            }

        }).saveAsTextFile("genOutput")

        // Need to remove temp files
        resData.foreach(r => {
            val tempFile = r.getString(0)
            println("Looking to delete " + tempFile)
            val tfile = new File(tempFile)
            if (tfile.exists()) {
                println("File deleted")
                tfile.delete()
            }
        })


        println("Creating Data Frame")

        val dFrame = sqlContext.createDataFrame(resData, schema)
        dFrame.registerTempTable("corners")
    

        // Construct an RDD of Rows containing only the fields we need. Cache the result
        println("Selection sql data")
        val input = sqlContext.sql("select cast(x0x as double), cast(x0y as double), cast(x1x as double), cast(x1y as double), cast(x2x as double), cast(x2y as double), cast(x3x as double), cast(x3y as double), type from corners")
          .rdd.cache()
        // add where type='c' for only corners.

        // Given an input row, return pickup longitude, latitude as a tuple
        val pickupExtractor = (r: Row) => {
          if (r.isNullAt(0) || r.isNullAt(1) || r.isNullAt(2) || r.isNullAt(3) ||
              r.isNullAt(4) || r.isNullAt(5) || r.isNullAt(6) || r.isNullAt(7)
             ) {
            None
          } else {
            Some((r.getDouble(0), r.getDouble(1), r.getDouble(2), r.getDouble(3),
                  r.getDouble(4), r.getDouble(5), r.getDouble(6), r.getDouble(7),
                  r.getString(8)
                  ))
          }
        }

        println("Creating TileGenerator")
        // Tile Generator object, which houses the generation logic
        val gen = TileGenerator(sc)

        // Break levels into batches. Process several higher levels at once because the
        // number of tile outputs is quite low. Lower levels done individually due to high tile counts.
        val levelBatches = List(List(0, 1, 2, 3, 4, 5, 6, 7, 8), List(9, 10, 11), List(12), List(13), List(14), List(15), List(16))

        // Iterate over sets of levels to generate.
        val levelMeta = levelBatches.map(level => {

          println("------------------------------")
          println(s"Generating level $level")
          println("------------------------------")

          // Construct the definition of the tiling jobs: pickups
          val pickups = new Series((tileSize - 1, tileSize - 1),
            pickupExtractor,
            new MercatorRectProjection(1024, level),
            (r: Row) => {
                   if (r.getString(8) == "corner"){
                       Some(1)
                   }else {
                       Some(1000)
                   }
            },
            CountAggregator,
            Some(MinMaxAggregator))

          // Create a request for all tiles on these levels, generate
          val request = new TileLevelRequest(level, (coord: (Int, Int, Int)) => coord._1)
          println("Generating tiles.")
          val rdd = gen.generate(input, pickups, request)
          // Translate RDD of Tiles to RDD of (coordinate,byte array), collect to master for serialization
          val output = rdd
            .map(s => pickups(s).get)
            .map(tile => {
              // Return tuples of tile coordinate, byte array
              (tile.coords, createByteBuffer(tile))
            })
            .collect()


          println("Writing out new tiles")
          // Save byte files to local filesystem
          output.foreach(tile => {
            val coord = tile._1
            val byteArray = tile._2
            val limit = (1 << coord._1) - 1
            // Use standard TMS path structure and file naming
            val file = new File(s"$outputPath/$layerName/${coord._1}/${coord._2}/${limit - coord._3}.bins")
            file.getParentFile.mkdirs()
            val output = new FileOutputStream(file)
            output.write(byteArray)
            output.close()
          })

          println("Creating map for each level")
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

        println("Saving meta.json file")
        // Flatten array of maps into a single map
        val levelInfoJSON = JSONObject(levelMeta.reduce(_ ++ _)).toString()
        // Save level metadata to filesystem
        val pw = new PrintWriter(s"$outputPath/$layerName/meta.json")
        pw.write(levelInfoJSON)
        pw.close()
    }

    sc.stop()

  }
}
