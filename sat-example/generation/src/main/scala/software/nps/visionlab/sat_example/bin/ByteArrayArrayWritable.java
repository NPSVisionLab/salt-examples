package software.nps.visionlab.sat_example.bin;

import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.BytesWritable;

/**
 * Created by trbatcha on 3/11/16.
 */
public class ByteArrayArrayWritable extends ArrayWritable{
    public ByteArrayArrayWritable(){
        super(BytesWritable.class);
    }
    public ByteArrayArrayWritable(BytesWritable[] values){
        super(BytesWritable.class, values);
    }
    public long getTotalLength() {
        BytesWritable bytes[] = (BytesWritable[])get();
        long len = 0;
        int i;
        for (i = 0; i < bytes.length; i++) {
            len += bytes[i].getLength();
        }
        return len;
    }
}
