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
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Vector;
import java.util.Properties;
import java.lang.IllegalArgumentException;
import java.lang.RuntimeException;

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

    private Properties lastProps;

    public static java.awt.geom.Point2D[] computeCornersFromGeotransform(double[] gt, int width, int height){
        if (null == gt || gt.length != GT_SIZE)
            return null;

        if (gt[GT_5_PIXEL_HEIGHT] > 0)
            gt[GT_5_PIXEL_HEIGHT] = -gt[GT_5_PIXEL_HEIGHT];


        java.awt.geom.Point2D[] corners = new java.awt.geom.Point2D[4];
        int i = 0;
        corners[i++] = getGeoPointForRasterPoint(gt, 0, height);
        corners[i++] = getGeoPointForRasterPoint(gt, width, height);
        corners[i++] = getGeoPointForRasterPoint(gt, width, 0);
        corners[i++] = getGeoPointForRasterPoint(gt, 0, 0);

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



public static java.awt.geom.Point2D[] getBoundingBox(SpatialReference srs, Dataset ds) throws IllegalArgumentException {
        java.awt.geom.Point2D[] bbox = null;
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
                //String EPSG4326WKT = "GEOGCS[\"WGS84 datum, Latitude-Longitude; Degrees\", DATUM[\"WGS_1984\", SPHEROID[\"World Geodetic System of 1984, GEM 10C\",6378137,298.257223563, AUTHORITY[\"EPSG\",\"7030\"]], AUTHORITY[\"EPSG\",\"6326\"]], PRIMEM[\"Greenwich\",0], UNIT[\"degree\",0.0174532925199433], AUTHORITY[\"EPSG\",\"4326\"]]";
                //SpatialReference newSpatialReference = new SpatialReference(EPSG4326WKT);

                String message = "Spatial Ref is null and can't get one from dataset";
                System.out.println(message);
                throw new IllegalArgumentException(message);

            }
        }

        if (srs.IsGeographic() == 0 && srs.IsProjected() == 0){
            String message = "Unprojected Coordinate System " + srs.ExportToWkt();
            System.out.println(message);
            throw new IllegalArgumentException(message);
        }

        double[] gt = new double[6];
        ds.GetGeoTransform(gt);
        java.awt.geom.Point2D[] corners = computeCornersFromGeotransform(gt, ds.getRasterXSize(), ds.getRasterYSize());
        bbox = calcBoundingSector(srs, corners);
        return bbox;
    }

    GDALFootprint() {
        System.out.println("GDAL init...");
        gdal.AllRegister();
    }

    public Properties getLastProps() {
        return lastProps;
    }



    public Dataset openFile(File f)
    {
        Dataset poDataset = null;
        try
        {
            System.out.println("trying to open " + f.getAbsolutePath());
            poDataset = (Dataset) gdal.Open(f.getAbsolutePath(),
                gdalconst.GA_ReadOnly);
            if (poDataset == null)
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
        lastProps = new Properties();
        lastProps.setProperty("name", f.getName());

        int width = poDataset.getRasterXSize();
        int height = poDataset.getRasterXSize();

        String projRef = poDataset.GetProjectionRef();

        if (poDataset.GetProjectionRef() != null)
            System.out.println("Projection is `" + poDataset.GetProjectionRef()
                + "'");

        Hashtable dict = poDataset.GetMetadata_Dict("");
        Enumeration keys = dict.keys();
        System.out.println(dict.size() + " items of metadata found (via Hashtable dict):");
        while (keys.hasMoreElements())
        {
            String key = (String) keys.nextElement();
            System.out.println(" :" + key + ":==:" + dict.get(key) + ":");
        }

        Vector list = poDataset.GetMetadata_List("");
        Enumeration enumerate = list.elements();
        System.out.println(list.size() + " items of metadata found (via Vector list):");
        while (enumerate.hasMoreElements())
        {
            String s = (String) enumerate.nextElement();
            System.out.println(" " + s);
        }



        poDataset.GetGeoTransform(adfGeoTransform);
        {
            System.out.println("Origin = (" + adfGeoTransform[0] + ", "
                + adfGeoTransform[3] + ")");

            System.out.println("Pixel Size = (" + adfGeoTransform[1] + ", "
                + adfGeoTransform[5] + ")");
            lastProps.setProperty("xpixsize", String.valueOf(adfGeoTransform[1]));
            lastProps.setProperty("ypixsize", String.valueOf(adfGeoTransform[5]));
        }


        int bandCount = poDataset.getRasterCount();
        lastProps.setProperty("bands", String.valueOf(bandCount));

        return poDataset;
    }

    public void printLastError()
    {
        System.out.println("Last error: " + gdal.GetLastErrorMsg());
        System.out.println("Last error no: " + gdal.GetLastErrorNo());
        System.out.println("Last error type: " + gdal.GetLastErrorType());
    }

    public DataBuffer getDataBuffer(Dataset ds){
        Band poBand = null;
        double[] adfMinMax = new double[2];
        Double[] max = new Double[1];
        Double[] min = new Double[1];

        int bandCount = ds.getRasterCount();
        ByteBuffer[] bands = new ByteBuffer[bandCount];
        int[] banks = new int[bandCount];
        int[] offsets = new int[bandCount];

        int xsize = 1024;//poDataset.getRasterXSize();
        int ysize = 1024;//poDataset.getRasterYSize();
        int pixels = xsize * ysize;
        int buf_type = 0, buf_size = 0;

        for (int band = 0; band < bandCount; band++)
        {
            /* Bands are not 0-base indexed, so we must add 1 */
            poBand = ds.GetRasterBand(band + 1);

            buf_type = poBand.getDataType();
            buf_size = pixels * gdal.GetDataTypeSize(buf_type) / 8;

            System.out.println(" Data Type = "
                + gdal.GetDataTypeName(poBand.getDataType()));
            System.out.println(" ColorInterp = "
                + gdal.GetColorInterpretationName(poBand
                .GetRasterColorInterpretation()));

            System.out.println("Band size is: " + poBand.getXSize() + "x"
                + poBand.getYSize());

            poBand.GetMinimum(min);
            poBand.GetMaximum(max);
            if (min[0] != null || max[0] != null)
            {
                System.out.println("  Min=" + min[0] + " Max="
                    + max[0]);
            }
            else
            {
                System.out.println("  No Min/Max values stored in raster.");
            }

            if (poBand.GetOverviewCount() > 0)
            {
                System.out.println("Band has " + poBand.GetOverviewCount()
                    + " overviews.");
            }

            if (poBand.GetRasterColorTable() != null)
            {
                System.out.println("Band has a color table with "
                    + poBand.GetRasterColorTable().GetCount() + " entries.");
                for (int i = 0; i < poBand.GetRasterColorTable().GetCount(); i++)
                {
                    System.out.println(" " + i + ": " +
                        poBand.GetRasterColorTable().GetColorEntry(i));
                }
            }

            System.out.println("Allocating ByteBuffer of size: " + buf_size);

            ByteBuffer data = ByteBuffer.allocateDirect(buf_size);
            data.order(ByteOrder.nativeOrder());

            int returnVal = 0;
            try
            {
                returnVal = poBand.ReadRaster_Direct(0, 0, poBand.getXSize(),
                    poBand.getYSize(), xsize, ysize,
                    buf_type, data);
            }
            catch (Exception ex)
            {
                System.err.println("Could not read raster data.");
                System.err.println(ex.getMessage());
                ex.printStackTrace();
                return null;
            }
            if (returnVal == gdalconstConstants.CE_None)
            {
                bands[band] = data;
            }
            else
            {
                printLastError();
            }
            banks[band] = band;
            offsets[band] = 0;
        }

        DataBuffer imgBuffer = null;
        SampleModel sampleModel = null;
        int data_type = 0, buffer_type = 0;

        if (buf_type == gdalconstConstants.GDT_Byte)
        {
            byte[][]bytes = new byte[bandCount][];
            for (int i = 0; i < bandCount; i++)
            {
                bytes[i] = new byte[pixels];
                bands[i].get(bytes[i]);
            }
            imgBuffer = new DataBufferByte(bytes, pixels);
            buffer_type = DataBuffer.TYPE_BYTE;
            sampleModel = new BandedSampleModel(buffer_type,
                xsize, ysize, xsize, banks, offsets);
            data_type = (poBand.GetRasterColorInterpretation() ==
                gdalconstConstants.GCI_PaletteIndex) ?
                BufferedImage.TYPE_BYTE_INDEXED : BufferedImage.TYPE_BYTE_GRAY;
        }
        else if (buf_type == gdalconstConstants.GDT_Int16)
        {
            short[][] shorts = new short[bandCount][];
            for (int i = 0; i < bandCount; i++)
            {
                shorts[i] = new short[pixels];
                bands[i].asShortBuffer().get(shorts[i]);
            }
            imgBuffer = new DataBufferShort(shorts, pixels);
            buffer_type = DataBuffer.TYPE_USHORT;
            sampleModel = new BandedSampleModel(buffer_type,
                xsize, ysize, xsize, banks, offsets);
            data_type = BufferedImage.TYPE_USHORT_GRAY;
        }
        else if (buf_type == gdalconstConstants.GDT_Int32)
        {
            int[][] ints = new int[bandCount][];
            for (int i = 0; i < bandCount; i++)
            {
                ints[i] = new int[pixels];
                bands[i].asIntBuffer().get(ints[i]);
            }
            imgBuffer = new DataBufferInt(ints, pixels);
            buffer_type = DataBuffer.TYPE_INT;
            sampleModel = new BandedSampleModel(buffer_type,
                xsize, ysize, xsize, banks, offsets);
            data_type = BufferedImage.TYPE_CUSTOM;
        }
        return imgBuffer;
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


