package software.nps.visionlab.sat_example.bin;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.ContentSummary;
import org.apache.hadoop.io.*;
import org.apache.hadoop.hdfs.DFSInputStream;
import org.apache.hadoop.hdfs.DFSInputStream.ReadStatistics;
import org.apache.commons.io.FilenameUtils;

import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.yarn.webapp.hamlet.HamletSpec;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;



/**
 * Created by trbatcha on 3/10/16.
 */
public class WholeFileRecordReader extends RecordReader<String, BytesWritable> {

    private FileSplit fileSplit;
    private Configuration conf;
    private TaskAttemptContext context;
    private boolean processed = false;
    private String key;
    private BytesWritable value = new BytesWritable();
    private final static int BUFF_SIZE = 8192;

    @Override
    public void initialize(InputSplit split, TaskAttemptContext context) throws IOException, InterruptedException{
        this.fileSplit = (FileSplit) split;
        this.conf = context.getConfiguration();
        this.context = context;
    }

    @Override
    public boolean nextKeyValue() throws IOException, InterruptedException {
        if (!processed){
            long llen = fileSplit.getLength();
            System.out.println("Next file split is " + String.valueOf(llen));
            Path file = fileSplit.getPath();
            FileSystem fs = file.getFileSystem(conf);
            value.set(new BytesWritable());
            String baseName = FilenameUtils.getName(file.getName());
            String tname = "/dev/shm/" + baseName;
            fs.copyToLocalFile(file, new Path(tname));
            key = tname;
            /*
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            byte[] temp = new byte[BUFF_SIZE];
            if (fs.isFile(file) == false){
               processed = true;
               return false;
            }
            FSDataInputStream in = null;
            try {
                System.out.println("opening file");
                in = fs.open(file);
                while(true) {
                    int res = 0;
                    try {
                        res = in.read(temp, 0, BUFF_SIZE); 
                    }catch(Exception e){
                        System.out.println("Exception");
                        break;
                    }
                    if (res > 0) {
                        bos.write(temp, 0, res);
                    }else {
                        break;
                    }
                }
                System.out.println("Writing bytes writable");
                value.set(new BytesWritable(bos.toByteArray()));
            } catch (IOException e) {
                System.out.println("IOException for file " + file.toString() +
                               " Error: "+ e.getMessage()); 
            } finally {
                IOUtils.closeStream(in);
            }
            */
            processed = true;
            return true;
        }
        return false;
    }

    @Override
    public String getCurrentKey() throws IOException, InterruptedException {
        //return fileSplit.getPath().getName();
        return key;
    }

    @Override
    public BytesWritable getCurrentValue() throws IOException, InterruptedException {
        return value;
    }

    @Override
    public float getProgress() throws IOException {
        return processed ? 1.0f : 0.0f;
    }

    @Override
    public void close() throws IOException {

    }
}
