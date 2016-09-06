/*   
 * (c) 2016 Rogue Wave Software, Inc.
 * This program is an example referenced by a blog post on SVM functionality
 * in JMSL v7.3. The JMSL Library is required to compile and run this
 * this code. The URL of blog post is:
 * http://blog.klocwork.com/featured/support-vector-machines-in-jmsl-part-2/
 * More information can be found at http://www.roguewave.com
 */

import javax.swing.*;
import java.awt.Font;
import java.awt.GridLayout;

import com.imsl.chart.*;
import com.imsl.datamining.supportvectormachine.*;

/* A textbook linearly separable example */
public class SVMnonlin {

    public static void main(String[] args) throws Exception {

        SVClassification.VariableType[] varType = {
            SVClassification.VariableType.CATEGORICAL,
            SVClassification.VariableType.QUANTITATIVE_CONTINUOUS,
            SVClassification.VariableType.QUANTITATIVE_CONTINUOUS
        };

        int N = 16;
        double[][] x = {
        		{0.0, 3.4, 4.1},
        		{0.0, 3.6, 5.9},
        		{0.0, 4.2, 5.0},
        		{0.0, 4.8, 7.0},
        		{0.0, 5.1, 3.1},
        		{0.0, 5.9, 5.8},
        		{0.0, 6.4, 4.0},
        		{0.0, 7.0, 6.1},
        		{1.0, 1.6, 6.2},
        		{1.0, 2.1, 3.1},
        		{1.0, 2.6, 1.8},
        		{1.0, 3.6, 8.2},
        		{1.0, 6.2, 1.0},
        		{1.0, 6.1, 8.4},
        		{1.0, 8.1, 3.4},
        		{1.0, 8.5, 7.1}
        };

        // Extract the known classification
        double[] knownClass = new double[N];
        for (int i=0; i<N; i++) {
            knownClass[i] = x[i][0];
        }

        // Construct a Support Vector Machine
        SVClassification svm = new SVClassification(x, 0, varType);

        // Use a 2nd order polynomial kernel function.
        svm.setKernel(new PolynomialKernel(1.0, 1.0, 2));

        // Train the model on the training sample
        svm.fitModel();

        // Get the fitted values (classify the training data)
        double[] fittedClass = svm.predict();
        new com.imsl.math.PrintMatrix("fittedClass").print(fittedClass);        
        int[][] fittedClassErrors = svm.getClassErrors(knownClass, fittedClass);
        new com.imsl.math.PrintMatrix("fittedClassErrors").print(fittedClassErrors);
        
        // classify on the diagonal as a test
        final int M = 16;
        final double step = 10.0/M;
        double[][] testLine = new double[M][3];
        for (int i=0; i<M; i++) {
        	testLine[i][1] = 0.3125 + i*step;
        	testLine[i][2] = 0.3125 + i*step;
        }
        double[] predictLine = svm.predict(testLine);
        new com.imsl.math.PrintMatrix("predict line").print(predictLine);

        /* Everything below here is for charting output */
        
        // group testline data for plotting
        double suml = 0;
        for (int i=0; i<M; i++) {
        	suml += predictLine[i];
        }
        int n1 = new Double(suml).intValue();	int n0 = M-n1;
        double[][] tl0 = new double[2][n0];
        int ct = 0;
        for (int i=0; i<M; i++) {
        	if (predictLine[i] == 0) {
        		tl0[0][ct] = testLine[i][1];
        		tl0[1][ct] = testLine[i][2];
        		ct++;
        	}
        }
        
        double[][] tl1 = new double[2][n1];
        ct = 0;
        for (int i=0; i<M; i++) {
        	if (predictLine[i] == 1) {
        		tl1[0][ct] = testLine[i][1];
        		tl1[1][ct] = testLine[i][2];
        		ct++;
        	}
        }        
        
        // Classify across domain for plotting
        double[][] test = new double[M*M][3];
        for (int i=0; i<M; i++) {
        	for (int j=0; j<M; j++) {
        		test[j+i*M][1] = i;
        		test[j+i*M][2] = j;
        	}
        }
        double[] predicted = svm.predict(test);
        
        double[][] raw0 = new double[2][N/2];
        double[][] raw1 = new double[2][N/2];
        for (int i=0; i<N/2; i++) {
        	for (int j=0; j<2; j++) {
        		raw0[j][i] = x[i][j+1];
        		raw1[j][i] = x[i+N/2][j+1];
        	}
        }
        
        double[][] z = new double[M][M];
        double[] g = new double[M];
        for (int i=0; i<M; i++) {
        	g[i] = i;
        	for (int j=0; j<M; j++) {
        		z[i][j] = predicted[j + i*M];
        	}
        }        
        
        // configure the Frame to hold both charts
        final int sz = 640;
        JFrame jf = new JFrame();
        jf.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        jf.setSize(sz*2,sz);
        jf.setLayout(new GridLayout(1,2));
        JPanelChart jp1 = new JPanelChart();
        jp1.setSize(sz,sz);
        JPanelChart jp2 = new JPanelChart();
        jp2.setSize(sz,sz);
        jf.add(jp1);
        jf.add(jp2);
       
        // first chart showing data
        Chart c = jp1.getChart();
        c.setFontSize(14);
        c.setFontStyle(Font.BOLD);
        AxisXY a = new AxisXY(c);
        a.setAutoscaleInput(Axis.AUTOSCALE_OFF);
        a.getAxisX().setWindow(new double[] {0, 10});
        a.getAxisY().setWindow(new double[] {0, 10});  
        a.getAxisX().setNumber(6);
        a.getAxisY().setNumber(6);
        a.getAxisX().setTextFormat("0");
        a.getAxisY().setTextFormat("0");
        a.getAxisX().getAxisTitle().setTitle("X");
        a.getAxisY().getAxisTitle().setTitle("Y");
        Data d0 = new Data(a, raw0[0], raw0[1]);
        Data d1 = new Data(a, raw1[0], raw1[1]);
        d0.setDataType(Data.DATA_TYPE_MARKER);
        d0.setMarkerColor(java.awt.Color.BLUE);
        d0.setMarkerType(Data.MARKER_TYPE_HOLLOW_CIRCLE);
        d0.setMarkerThickness(2.0);
        d0.setTitle("0");
        d1.setMarkerColor(java.awt.Color.RED);
        d1.setDataType(Data.DATA_TYPE_MARKER);
        d1.setMarkerType(Data.MARKER_TYPE_PLUS);
        d1.setMarkerThickness(2.0);
        d1.setTitle("1");
        c.getLegend().setPaint(true);
        c.getLegend().setTitle("Legend");
        c.getLegend().setViewport(0.225, 0.325, 0.125, 0.325);       
        
        // second chart showing classification of a test line    
        Chart c2 = jp2.getChart();
        c2.setFontSize(14);
        c2.setFontStyle(Font.BOLD);        
        AxisXY a2 = new AxisXY(c2);
        a2.getAxisX().setTextFormat("0");
        a2.getAxisY().setTextFormat("0");
        a2.getAxisX().getAxisTitle().setTitle("X");
        a2.getAxisY().getAxisTitle().setTitle("Y");        
        Data d20 = new Data(a2, raw0[0], raw0[1]);
        Data d21 = new Data(a2, raw1[0], raw1[1]);
        d20.setDataType(Data.DATA_TYPE_MARKER);
        d20.setMarkerColor(java.awt.Color.BLUE);
        d20.setMarkerType(Data.MARKER_TYPE_HOLLOW_CIRCLE);
        d20.setMarkerThickness(2.0);
        d20.setTitle("0");
        d21.setMarkerColor(java.awt.Color.RED);
        d21.setDataType(Data.DATA_TYPE_MARKER);
        d21.setMarkerType(Data.MARKER_TYPE_PLUS);
        d21.setMarkerThickness(2.0);        
        d21.setTitle("1");
        Data d20t = new Data(a2, tl0[0], tl0[1]);
        Data d21t = new Data(a2, tl1[0], tl1[1]);
        d20t.setDataType(Data.DATA_TYPE_MARKER);
        d20t.setMarkerColor(java.awt.Color.BLUE);
        d20t.setMarkerType(Data.MARKER_TYPE_FILLED_SQUARE);
        d20t.setMarkerSize(0.75);
        d20t.setTitle("Classified 0");
        d21t.setMarkerColor(java.awt.Color.RED);
        d21t.setDataType(Data.DATA_TYPE_MARKER);
        d21t.setMarkerType(Data.MARKER_TYPE_FILLED_SQUARE);        
        d21t.setMarkerSize(0.75);
        d21t.setTitle("Classified 1");
        c2.getLegend().setPaint(true);
        c2.getLegend().setTitle("Legend");
        c2.getLegend().setViewport(0.225, 0.325, 0.125, 0.325);
        
        jf.setVisible(true);
    }
}
