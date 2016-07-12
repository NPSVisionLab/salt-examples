/******************************************************************************
 * $Id$
 *
 * Name:     GDALFootprint.java
 * Author:   Thomas Batcha
 * 
 *
 * $Log$
 * Revision 1.1  2006/02/08 19:39:03  collinsb
 * Initial version
 *
 *
 */

package software.nps.visionlab.sat_example.bin;
import java.awt.BorderLayout;
import java.awt.color.ColorSpace;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BandedSampleModel;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.ComponentColorModel;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.awt.image.DataBufferShort;
import java.awt.image.Raster;
import java.awt.image.SampleModel;
import java.awt.image.WritableRaster;
import java.awt.geom.Point2D;
import java.io.File;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Vector;
import java.util.ArrayList;
import java.util.Properties;
import java.util.StringTokenizer;
import java.lang.IllegalArgumentException;
import java.lang.RuntimeException;
import java.lang.Process;

import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.gdal.gdal.Band;
import org.gdal.gdal.Dataset;
import org.gdal.gdal.Driver;
import org.gdal.gdal.GCP;
import org.gdal.gdal.gdal;
import org.gdal.gdalconst.gdalconst;
import org.gdal.gdalconst.gdalconstConstants;
import org.gdal.osr.*;

import java.awt.geom.*;


public class GDALFootprint
{
    public static final int GT_SIZE = 6;

    public static final int GT_0_ORIGIN_LON = 0;
    public static final int GT_1_PIXEL_WIDTH = 1;
    public static final int GT_2_ROTATION_X = 2;
    public static final int GT_3_ORIGIN_LAT = 3;
    public static final int GT_4_ROTATION_Y = 4;
    public static final int GT_5_PIXEL_HEIGHT = 5;

    private Properties _lastProps;
    private String _fileName;
    private Dataset _poDataset;
    private SpatialReference _srs = null;

    public GDALFootprint() {
        System.out.println("GDAL init...");
        gdal.AllRegister();
    }

    public static class DetectionResult {
        public String objName;
        public int width;
        public int height;
        public java.awt.geom.Point2D p0;
        public java.awt.geom.Point2D p1;
        public java.awt.geom.Point2D p2;
        public java.awt.geom.Point2D p3;
    }

    public static java.awt.geom.Point2D[] computeCornersFromGeotransform(
                    double[] gt, int x, int y, int width, int height){
        if (null == gt || gt.length != GT_SIZE)
            return null;

        if (gt[GT_5_PIXEL_HEIGHT] > 0)
            gt[GT_5_PIXEL_HEIGHT] = -gt[GT_5_PIXEL_HEIGHT];


        java.awt.geom.Point2D[] corners = new java.awt.geom.Point2D[4];
        int i = 0;
        corners[i++] = getGeoPointForRasterPoint(gt, x, y + height -1);
        corners[i++] = getGeoPointForRasterPoint(gt, x + width -1 , 
                                                     y + height -1);
        corners[i++] = getGeoPointForRasterPoint(gt, x + width -1, y);
        corners[i++] = getGeoPointForRasterPoint(gt, x, y);

        return corners;
    }

    public static java.awt.geom.Point2D getGeoPointForRasterPoint(double[] gt, int x, int y){
        java.awt.geom.Point2D geoPoint = null;

        if (null != gt && gt.length == 6)
        {
            double easting = gt[GT_0_ORIGIN_LON] + gt[GT_1_PIXEL_WIDTH] * (double) x
            + gt[GT_2_ROTATION_X] * (double) y;

            double northing = gt[GT_3_ORIGIN_LAT] + gt[GT_4_ROTATION_Y] * (double) x
            + gt[GT_5_PIXEL_HEIGHT] * (double) y;

            geoPoint = new java.awt.geom.Point2D.Double(easting, northing);
            }

        return geoPoint;
    }

    public static java.awt.geom.Point2D[] calcBoundingSector(SpatialReference srs, java.awt.geom.Point2D[] corners)
        throws IllegalArgumentException{
        if (null == srs)
        {
            String message = "nullValue.SpatialReferenceIsNull";
            System.out.println(message);
            throw new IllegalArgumentException(message);
        }

        if (null == corners)
        {
            String message = "nullValue.ArrayIsNull";
            System.out.println(message);
            throw new IllegalArgumentException(message);
        }

        java.awt.geom.Point2D[] bbox = new java.awt.geom.Point2D[4];
        try
        {

            //CoordinateTransformation ct = new CoordinateTransformation(srs, GDALUtils.createGeographicSRS());
            // Transform to WGS84 latlong
            SpatialReference dst = new SpatialReference("");
            dst.ImportFromProj4("+proj=latlong +datum=WGS84 +no_defs");
            CoordinateTransformation ct = new CoordinateTransformation(srs, dst);
            int i = 0;
            for (java.awt.geom.Point2D corner : corners) {
                double[] point = ct.TransformPoint(corner.getX(), corner.getY());
                bbox[i++] = new java.awt.geom.Point2D.Double(point[0], point[1]);
            }

        }
        catch (Throwable t)
        {
            String error = t.getMessage();
            String reason = (null != error && error.length() > 0) ? error : t.getMessage();
            String message = reason;
            System.out.println(message);
            throw new RuntimeException(message);
        }
        return bbox;
    }

    public static SpatialReference getSpatialReference(SpatialReference sp,
                               Dataset ds) 
                              throws IllegalArgumentException {
        SpatialReference srs = sp;
        if (null == ds){
            String message = "Data Set is NULL";
            System.out.println(message);
            throw new IllegalArgumentException(message);
        }
        if (null == srs){
            String wkt = ds.GetProjectionRef();
            if (null != wkt &&  wkt.length() > 0){
                 srs = new SpatialReference(wkt);
            }
            if (null == srs){
                String message = "Spatial Ref is null and can't get one from dataset";
                System.out.println(message);
                throw new IllegalArgumentException(message);
            }
        }
        if (srs.IsGeographic() == 0 && srs.IsProjected() == 0){
            String message = "Unexpected Coordinate System " + srs.ExportToWkt();
            System.out.println(message);
            throw new IllegalArgumentException(message);
        }
        return srs;
    }


    public  java.awt.geom.Point2D[] getBoundingBox(SpatialReference sp, Dataset ds) throws IllegalArgumentException {
        java.awt.geom.Point2D[] bbox = null;
        SpatialReference srs;
        if (_srs != null)
            srs = _srs;
        else{
            srs = getSpatialReference(sp, ds);
            _srs = srs;
        }
        double[] gt = new double[6];
        ds.GetGeoTransform(gt);
        java.awt.geom.Point2D[] corners = computeCornersFromGeotransform(gt, 0,
                                   0, ds.getRasterXSize(), ds.getRasterYSize());
        bbox = calcBoundingSector(srs, corners);
        return bbox;
    }

    public Properties getLastProps() {
        return _lastProps;
    }



    public Dataset openFile(File f)
    {
        _poDataset = null;
        try
        {
            System.out.println("trying to open " + f.getAbsolutePath());
            _fileName = f.getAbsolutePath();
            _poDataset = (Dataset) gdal.Open(_fileName, gdalconst.GA_ReadOnly);
            if (_poDataset == null)
            {
                System.out.println("The image could not be read.");
                printLastError();
                return null;
            }
        }
        catch (Exception e)
        {
            System.err.println("Exception caught.");
            System.err.println(e.getMessage());
            e.printStackTrace();
            return null;
        }
        double[] adfGeoTransform = new double[6];
        _lastProps = new Properties();
        _lastProps.setProperty("name", f.getName());

        int width = _poDataset.getRasterXSize();
        int height = _poDataset.getRasterXSize();

        String projRef = _poDataset.GetProjectionRef();

        if (_poDataset.GetProjectionRef() != null)
            System.out.println("Projection is `" + _poDataset.GetProjectionRef()
                + "'");

        Hashtable dict = _poDataset.GetMetadata_Dict("");
        Enumeration keys = dict.keys();
        System.out.println(dict.size() + " items of metadata found (via Hashtable dict):");
        while (keys.hasMoreElements())
        {
            String key = (String) keys.nextElement();
            System.out.println(" :" + key + ":==:" + dict.get(key) + ":");
        }

        Vector list = _poDataset.GetMetadata_List("");
        Enumeration enumerate = list.elements();
        System.out.println(list.size() + " items of metadata found (via Vector list):");
        while (enumerate.hasMoreElements())
        {
            String s = (String) enumerate.nextElement();
            System.out.println(" " + s);
        }



        _poDataset.GetGeoTransform(adfGeoTransform);
        {
            System.out.println("Origin = (" + adfGeoTransform[0] + ", "
                + adfGeoTransform[3] + ")");

            System.out.println("Pixel Size = (" + adfGeoTransform[1] + ", "
                + adfGeoTransform[5] + ")");
            _lastProps.setProperty("xpixsize", String.valueOf(adfGeoTransform[1]));
            _lastProps.setProperty("ypixsize", String.valueOf(adfGeoTransform[5]));
        }


        int bandCount = _poDataset.getRasterCount();
        _lastProps.setProperty("bands", String.valueOf(bandCount));

        return _poDataset;
    }
    public void CloseDataset() {
        if (_poDataset != null)
            _poDataset.delete();
    }

    public void printLastError()
    {
        System.out.println("Last error: " + gdal.GetLastErrorMsg());
        System.out.println("Last error no: " + gdal.GetLastErrorNo());
        System.out.println("Last error type: " + gdal.GetLastErrorType());
    }


    public java.awt.geom.Point2D[] getCorners(File f){
        java.awt.geom.Point2D[] points = null;
        Dataset ds = openFile(f);
        if (ds != null) {
            points =  getCorners(ds);
        }
        return points;
    }

    public java.awt.geom.Point2D[] getCorners(Dataset ds){
        java.awt.geom.Point2D[] points = null;
        // IF we have four control points use them as corners
        Vector GCPs = new Vector();
        ds.GetGCPs(GCPs);
        System.out.println("Got " + GCPs.size() + " GCPs");
        Enumeration e = GCPs.elements();
        if (GCPs.size() == 4) {
            int i = 0;
            points = new java.awt.geom.Point2D[4];
            while (e.hasMoreElements()) {
                GCP gcp = (GCP) e.nextElement();
                System.out.println(" x:" + gcp.getGCPX() +
                    " y:" + gcp.getGCPY()); //+
                //" z:" + gcp.getGCPZ() +
                //" pixel:" + gcp.getGCPPixel() +
                //" line:" + gcp.getGCPLine() +
                //" line:" + gcp.getInfo());
                points[i] = new java.awt.geom.Point2D.Double(gcp.getGCPX(), gcp.getGCPY());
                i++;
            }
        }
        if (points == null) {
            points = getBoundingBox(null, ds);
            if (points != null){
                int i;
                for (i = 0; i < 4; i++) {
                    System.out.println("X:" + String.valueOf(points[i].getX()) + " Y:" + String.valueOf(points[i].getY()));
                }
            }
        }
        return points;
    }

    public ArrayList<DetectionResult> getDetections(
                                      String detectorScript) throws
                                      InterruptedException, IOException{
        System.out.println("script " + detectorScript);
        ArrayList<DetectionResult> res = new ArrayList<DetectionResult>();
        if (new File(detectorScript).exists() == false){
            System.out.println("Detector script " + detectorScript + " does not exist!!!");
            return res;
        }
        if (new File("/usr/bin/python").exists() == false){
            System.out.println("/usr/bin/python does not exist!!!");
            return res;
        }
        ArrayList<String> args = new ArrayList<String>();
        args.add("/usr/bin/python");
        args.add(detectorScript);
        args.add("--cpu");
        args.add(_fileName);
        ProcessBuilder pb = new ProcessBuilder(args);
        pb.environment().put("LD_LIBRARY_PATH",
        "/usr/lib:/usr/lib64:/usr/lib64/atlas:/home/trbatcha/tools/lib:/home/trbatcha/tools/usr/lib64:/home/trbatcha/tools/usr/lib64/atlas:/home/trbatcha/tools/usr/lib:/home/trbatcha/gflags-2.1.1/build/lib:/home/trbatcha/liblmdb:/home/trbatcha/leveldb-master:/home/trbatcha/usr/lib:/home/trbatcha/tools/opencv/lib:/usr/lib64:/home/trbatcha/work/py-faster-rcnn/caffe-fast-rcnn/.build_release/lib");
        pb.environment().put("PYTHONPATH",
        "/home/trbatcha/work/py-faster-rcnn/lib:/home/trbatcha/work/py-faster-rcnn/caffe-fast-rcnn/python:$PYTHONPATH");

        Process process = pb.start();
        int errCode = process.waitFor();
        if (errCode != 0) {
            System.out.println("Could not execute detection script!!");
            System.out.println("Error code: " + String.valueOf(errCode));
            return res;
        }
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(
                                 process.getInputStream()));
            String line = null;
            // Input steam should be objname  x y w h
            // The x,y,w,h are in image pixel coordinates we
            // need to convert to lat lon
            String objName;
            while ((line = br.readLine()) != null) {
                System.out.println("reading next line from input stream");
                System.out.println(line);
                StringTokenizer tok = new StringTokenizer(line);
                DetectionResult dres = new DetectionResult();
                dres.objName = (String)tok.nextElement();
                System.out.println("objname " + dres.objName);
                dres.p0 = new Point2D.Double(
                                  Integer.valueOf((String)tok.nextElement()),
                                  Integer.valueOf((String)tok.nextElement()));
                dres.width = Integer.valueOf((String)tok.nextElement());
                dres.height = Integer.valueOf((String)tok.nextElement());
                System.out.println("adding to res " + String.valueOf(dres.width));
                res.add(dres);
            }
        } finally {
            br.close();
        }
        // Now convert the points to lan lon
        int i;
        SpatialReference srs;
        if (_srs == null){
            srs = getSpatialReference(null, _poDataset);
            _srs = srs;
        }else
            srs = _srs;
        double[] gt = new double[6];
        _poDataset.GetGeoTransform(gt);
        for (i = 0; i < res.size(); i++) {
            java.awt.geom.Point2D[] corners = 
              computeCornersFromGeotransform(gt, (int)res.get(i).p0.getX(),
               (int)res.get(i).p0.getY(), res.get(i).width, res.get(i).height);
            java.awt.geom.Point2D[] bbox = calcBoundingSector(srs, corners);
            DetectionResult dres = res.get(i);
            System.out.println("box " + 
                               String.valueOf(bbox[0].getX()) + "," +
                               String.valueOf(bbox[0].getY()) + "," +
                               String.valueOf(bbox[1].getX()) + "," +
                               String.valueOf(bbox[1].getY()) + "," +
                               String.valueOf(bbox[2].getX()) + "," +
                               String.valueOf(bbox[2].getY()) + "," +
                               String.valueOf(bbox[2].getX()) + "," +
                               String.valueOf(bbox[2].getY()));
            dres.p0 = bbox[0];
            dres.p1 = bbox[1];
            dres.p2 = bbox[2];
            dres.p3 = bbox[3];
            res.set(i,dres);
        }
        System.out.println("java returning res");
        System.out.println(String.valueOf(res.get(0).p0.getX()));
        return res;
    }

    /** @param args  */
    public static void main(String[] args)
    {
        System.out.println("GDAL init...");
        gdal.AllRegister();
        /*
        int count = gdal.GetDriverCount();
        System.out.println(count + " available Drivers");
        for (int i = 0; i < count; i++)
        {
            try
            {
                Driver driver = gdal.GetDriver(i);
                System.out.println(" " + driver.getShortName() + " : "
                    + driver.getLongName());
            }
            catch (Exception e)
            {
                System.err.println("Error loading driver " + i);
            }
        }
        */
        GDALFootprint fp = new GDALFootprint();
        if (args.length >= 1)
        {
            Dataset ds = fp.openFile(new File(args[0]));
            java.awt.geom.Point2D[] points = fp.getCorners(ds);

        }
    }
}


