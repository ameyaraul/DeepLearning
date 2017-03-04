/**
 * @Author: Yuting Liu and Jude Shavlik.  
 * 
 * Copyright 2017.  Free for educational and basic-research use.
 * 
 * The main class for Lab3 of cs638/838.
 * 
 * Reads in the image files and stores BufferedImage's for every example.  Converts to fixed-length
 * feature vectors (of doubles).  Can use RGB (plus grey-scale) or use grey scale.
 * 
 * You might want to debug and experiment with your Deep ANN code using a separate class, but when you turn in Lab3.java, insert that class here to simplify grading.
 * 
 * Some snippets from Jude's code left in here - feel free to use or discard.
 *
 */

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Vector;

import javax.imageio.ImageIO;




/**
 * Reperesents a single unit of a neural network
 * @author ameyaraul
 *
 */
class Perceptron {
	public double[] w;
	public double w_bias;
	public double eta;
	public int noEpochs;
	public double output;
	public double delta;
	public double delta_prev; // to add momentum
	public double alpha; //momentum term
	public static final int MAX_PATIENCE = 20;
	public static final int SEED = 8;
	
	public Instance[] train;
	public Instance[] tune;
	public Instance[] test;
	
	public Perceptron(){delta = 0;};
	
	public Perceptron(double eta, int size, double alpha, int seed){
		this.eta = eta;
		w = new double[size];
		//Random rand = new Random(SEED);
		Random rand = new Random(seed);
		w_bias = 0.02*rand.nextDouble() - 0.01;
		for (int i = 0; i < w.length; i++) {
			w[i] = 0.02*rand.nextDouble()-0.01;
		}
		delta = 0;
		this.alpha = alpha;
	}
	
	
	public double getwx(List<Double> list) {
		double sum = 0;
		for (int i = 0; i < list.size(); i++) {
			sum += w[i] * list.get(i);
		}
		return sum + w_bias * -1;
		
	}
	
	public void sigmoidUpdateWeights(double deltaw, List<Double> inputs) {
		delta_prev = delta;
		delta = output*(1-output)*deltaw;
		for (int i = 0; i < inputs.size(); i++) {
			w[i] += eta*inputs.get(i) * delta + alpha*delta_prev;
		}
		w_bias += eta*delta*-1 + alpha*delta_prev;
	}
	
	public double getSigmoidOutput(List<Double> values) {
		double wx = getwx(values);
		output = 1/(1 + Math.exp(-wx));
		if (output == 1) {
			System.out.println(Math.exp(-wx));
			System.out.println("God save my sorry soul!");
		}
		return output;
	}
	
	public double setSigmoidOutput(double wx) {
		output = 1.0/(1.0 + Math.exp(-wx));
		if (output == 1) {
			System.out.println(Math.exp(-wx));
			System.out.println("God save my sorry soul version 2!");
		}
		return output;
	}
}


abstract class Layer {
	int output_size;
	int input_size;
	Perceptron[] units;
	List<Integer> dropped_units;
	double eta;
	double alpha;
	double dropout;
	public ArrayList<Double> output;
	
	// Update outputs of the layer
	abstract public void updateOutput(List<Double> values);
	// Update weights if this is the output layer
	abstract public void updateWeights(double label, ArrayList<Double> layer_inputs);
	// Update weights if this is an intermediate/first layer
	abstract public void updateWeights(List<Double> layer_inputs, Layer nextLayer);
	// Get the delta.w for all units of the current layer attached to a unit of the previous layer
	abstract public double getDeltaW(int unit_index);
}

class DenseLayer extends Layer {
	
	public DenseLayer(int output_size, int input_size, double eta, double alpha, double dropout) {
		this.output_size = output_size;
		this.input_size = input_size;
		this.units = new Perceptron[output_size];
		this.eta = eta;
		this.alpha = alpha;
		this.dropout = dropout;
		
		for (int i = 0; i < output_size; i++) {
			units[i] = new Perceptron(eta, input_size, alpha, i);
		}
		
		output = new ArrayList<Double>(output_size);
		for (int i = 0; i < output_size; i++) {
			output.add(i, 0.0); 
		}
	}
	
	@Override
	public void updateOutput(List<Double> values){
		int noToDrop = (int)Math.floor(dropout*output_size);
		dropped_units = new ArrayList<Integer>(noToDrop);
		Random randGen = new Random(ANN.SEED);
		for (int j = 0; j < noToDrop; j++) {
			int unit_no = randGen.nextInt(noToDrop);
			if(dropped_units.contains(unit_no))
				j--;
			else {
				dropped_units.add(unit_no);
			}
		}

		for (int i = 0 ; i < output_size; i++) {
			if (dropped_units.contains(i)) {
				units[i].output = 0;
				output.set(i, 0.0);
				continue;
			}
			output.set(i, units[i].getSigmoidOutput(values));
		}
	}
	
//	public void updateEncoding(Vector<Double> values) {
//		for (int i = 0 ; i < output_size; i++) {
//			output.set(i, units[i].getSigmoidOutput(values));//.subList(i*input_size, (i + 1)* input_size)
//		}
//	}
	
	public void updateWeights(ArrayList<Double> labels, ArrayList<Double> layer_inputs) {
		for (int i = 0; i < output_size; i++) {
			if (dropped_units.contains(i)){
				continue;
			}
			units[i].sigmoidUpdateWeights(labels.get(i) - units[i].output, layer_inputs);
		}
	}
	
	@Override
	public void updateWeights(double label, ArrayList<Double> layer_inputs) {
		for (int i = 0; i < output_size; i++) {
			if (dropped_units.contains(i)){
				continue;
			}
			double y = 0;
			if (label == i) 
				y = 1;
			units[i].sigmoidUpdateWeights(y - units[i].output, layer_inputs);
		}
	}
	
	@Override
	public void updateWeights(List<Double> layer_inputs, Layer nextLayer) {
		for (int i = 0; i < output_size; i++) {
			if (dropped_units.contains(i)){
				continue;
			}
			// calculate delta*w 
			// unit[i] is connected to units in the nextLayer
			double deltaw = nextLayer.getDeltaW(i);
			units[i].sigmoidUpdateWeights(deltaw, layer_inputs);
		}
	}
	
	public double getDeltaW(int unit_index) {
		double deltaw = 0;
		for (int j = 0; j < this.output_size; j++) {
			deltaw += units[j].delta * units[j].w[unit_index];
		}
		return deltaw;
	}
	
//	public void updateEncodingWeights(Vector<Double> layer_inputs, Layer nextLayer) {
//		for (int i = 0; i < output_size; i++) {
//			// calculate delta*w 
//			// unit[i] is connected to units in the nextLayer
//			double deltaw = 0;
//			for (int j = 0; j < nextLayer.output_size; j++) {
//				deltaw += nextLayer.units[j].delta * nextLayer.units[j].w[i];
//			}
//			units[i].sigmoidUpdateWeights(deltaw, layer_inputs);//.subList(i*input_size, (i + 1)*input_size)
//		}
//	}
}

class ConvolutionLayer extends Layer {
	
	double[] w;
	double w_bias;
	int dimensions;
	int convWindowSize;
	int outputSideDim;
	int noPlates;
	int inputImageSize;
	int unitsPerPlate;
	
	public ConvolutionLayer(int no_plates, int inputImageSize, int conv_window_size, int dimension, double eta, double alpha, double dropout) {
		this.inputImageSize = inputImageSize;
		this.noPlates = no_plates;
		this.outputSideDim = inputImageSize - 2*(int)Math.floor(conv_window_size/2);
		this.output_size = outputSideDim * outputSideDim * no_plates;
		this.convWindowSize = conv_window_size;
		this.units = new Perceptron[output_size];
		this.eta = eta;
		this.alpha = alpha;
		this.dropout = dropout;
		this.dimensions = dimension;
		this.unitsPerPlate = outputSideDim * outputSideDim;
		
		w = new double[conv_window_size * conv_window_size * noPlates * dimensions];
		//Random rand = new Random(SEED);
		Random rand = new Random(ANN.SEED);
		w_bias = 0.02*rand.nextDouble() - 0.01;
		for (int i = 0; i < w.length; i++) {
			w[i] = 0.02*rand.nextDouble()-0.01;
		}
		
		for (int i = 0; i < output_size; i++) {
			units[i] = new Perceptron();
			units[i].w = w;
			units[i].w_bias = w_bias;

		}
		
		output = new ArrayList<Double>(output_size);
		for (int i = 0; i < output_size; i++) {
			output.add(i, 0.0); 
		}
	}

	@Override
	public void updateOutput(List<Double> values) {
		// TODO: Add dropout!!
		// Our output vector is formatted as follows
		// o1p1 o1p2 o1p3 .... o2p1 o2p2 o2p3 ... p3p1 o3p2 o3p3 ...
		// The w vector is also stored the same way
		// w1p1d1 w1p1d2 w1p1s3 ... w1p2d1 w1p2d2 w1p2d3... w2p1d1 w2p1d2 w2p2d2...
		for (int i = 0; i < outputSideDim; i++) {
			for(int j = 0; j < outputSideDim; j++) {
				for(int plate = 0; plate < noPlates; plate++){
					// For each plate compute the outputs
					// we are looking at unit (i,j) in the output of plate 'plate' 
					// lets collect the weighted sum of corresponding input
					// for every (i,j) in the output, the corresponding position in input is (i+window/2, j + window/2)
					// So now loop over all those positions and compute the weighted sum 
					
					double wx = 0;
					for (int k = i; k < i + convWindowSize; k++) {
						for(int l = j; l < j + convWindowSize; l++) {
							// First covert 2-D indexing to 1-D
							int w_index = (((k-i)*this.convWindowSize + (l-j))*noPlates + plate)*dimensions;
							// Note that (k,l) is the index in the input image
							int image_index = (k)*this.inputImageSize + (l);
							
							// Now for all the input dimensions e.g dim=4 for color compute w.x
							for (int offset = 0; offset < dimensions; offset++) {
								wx += w[w_index + offset] * values.get(image_index + offset);
							}
							
						}	
					}
					// Add bias
					wx += w_bias*-1;
					
					// Now set the units output to the correct value
					// Currently using SIGMOID
					// First find the index of the unit in the units array
					int unit_index = (i*outputSideDim + j)*noPlates + plate;
					output.set(unit_index, units[unit_index].setSigmoidOutput(wx));				
				}
			}
		}
		
		
	}

	@Override
	public void updateWeights(double label, ArrayList<Double> layer_inputs) {
		// Highly unlikely we are ever gonna use this one as the Conv layer hardly appears as the output layer
		// of the ANN
		
	}

	@Override
	public void updateWeights(List<Double> layer_inputs, Layer nextLayer) {
		double[] netDelta = new double[w.length];
		double netDeltaBias = 0;
		for (int i = 0; i < output_size; i++) {
			// TODO: Fix this when we add dropout
//			if (dropped_units.contains(i)){
//				continue;
//			}
			// calculate delta*w 
			// unit[i] is connected to units in the nextLayer
			double deltaw = nextLayer.getDeltaW(i);
			units[i].delta = output.get(i)*(1-output.get(i))*deltaw;
			
			int plate = i%noPlates;
			int x = (i/noPlates)/outputSideDim;
			int y = (i/noPlates)%outputSideDim;
			
			for (int k = x; k < x + convWindowSize; k++) {
				for (int l = y; l < y + convWindowSize; l++) {
					// First covert 2-D indexing to 1-D
					int w_index = (((k-x)*this.convWindowSize + (l-y))*noPlates + plate)*dimensions;
					// Note that (k,l) is the index in the input image
					int image_index = (k)*this.inputImageSize + (l);
					// Now for all the input dimensions e.g dim=4 for color compute w.x
					for (int offset = 0; offset < dimensions; offset++) {
						netDelta[w_index + offset] += eta*layer_inputs.get(image_index) * units[i].delta;
					}
				}
			}
			netDeltaBias += units[i].delta;
			
		}
		for (int i = 0; i < w.length; i++) {
			w[i] += netDelta[i];
		}
		w_bias += eta*netDeltaBias*-1;
	}

	@Override
	public double getDeltaW(int unit_index) {
		// First convert unit_index to 2D coordinates
		int x = (unit_index/dimensions) / inputImageSize;
		int y = (unit_index/dimensions) % inputImageSize;
		int dim = unit_index%dimensions;
		double deltaw = 0;
		// So this pixel influences the output of units from 
		// (x-window + 1,y-window + 1) to (x, y) for each plate
		for (int plate = 0; plate < noPlates; plate++) {
			for (int i = x - convWindowSize + 1; i <= x; i++) {
				if (!(i >=0 && i <outputSideDim))
					continue;
				for (int j = y - convWindowSize + 1; j <= y; j++) {
					if (!(j>=0 && j <outputSideDim))
						continue;
					deltaw += w[(((x-i)*this.convWindowSize + (y-j))*noPlates + plate)*dimensions + dim] * units[(i*outputSideDim + j)*noPlates + plate].delta;
				}
			}
		}
		return deltaw;
	}
}

class MaxPoolingLayer extends Layer {

	int inputDimensions;
	int windowSize;
	int windowStride;
	int outputSideDim;
	int noPlates;
	int inputImageSize;
	int unitsPerPlate;
	List<Double> inputValues;
	Layer nextLayer;
	
	public MaxPoolingLayer(int no_plates, int inputImageSize, int window_size, int window_stride, double eta, double alpha, double dropout) {
		this.inputImageSize = inputImageSize;
		this.noPlates = no_plates;
		this.windowSize = window_size;
		this.windowStride = window_stride;
		
		this.outputSideDim = (inputImageSize - 2*(int)Math.floor(window_size/2))/window_stride + 1;
		this.output_size = outputSideDim * outputSideDim * no_plates;
		
		this.eta = eta;
		this.alpha = alpha;
		this.dropout = dropout;
		// input and output dimensions must be equal
		// e.g. if you have 20 plates on input, you have 20 corresponding 1-1 mapped max pool plates
		this.inputDimensions = no_plates; 
		this.unitsPerPlate = outputSideDim * outputSideDim;
		
		output = new ArrayList<Double>(output_size);
		for (int i = 0; i < output_size; i++) {
			output.add(i, 0.0); 
		}
	}
	@Override
	public void updateOutput(List<Double> values) {
		// Storing the input values for referencing later
		// so we can access it in the getDeltaw method
		this.inputValues = values;
		
		// TODO: Add dropout!!
		// Our output vector is formatted as follows
		// o1p1 o1p2 o1p3 .... o2p1 o2p2 o2p3 ... p3p1 o3p2 o3p3 ...
		for (int i = 0; i < outputSideDim; i++) {
			for(int j = 0; j < outputSideDim; j++) {
				for(int plate = 0; plate < noPlates; plate++){
					// For each plate compute the outputs
					// we are looking at unit (i,j) in the output of plate 'plate' 
					// lets collect the weighted sum of corresponding input
					// for every (i,j) in the output, the corresponding position in input is (i+window/2, j + window/2)
					// So now loop over all those positions and compute the weighted sum 
					
					double max = 0;
					for (int k = i; k < i + windowSize; k++) {
						for(int l = j; l < j + windowSize; l++) {
							// First covert 2-D indexing to 1-D
							// Note that (k,l) is the index in the input image
							int image_index = ((k)*this.inputImageSize + (l))*inputDimensions;
							
							if (values.get(image_index + plate) > max) {
								max = values.get(image_index + plate);
							}
							
						}	
					}
					
					
					// Now set the units output to the correct value
					// Currently using SIGMOID
					// First find the index of the unit in the units array
					int unit_index = (i*outputSideDim + j)*noPlates + plate;
					output.set(unit_index, max);				
				}
			}
		}
		
	}

	@Override
	public void updateWeights(double label, ArrayList<Double> layer_inputs) {
		// No waits to update
		// Also do not use this layer as the output layer
		// getDeltaW will give a null pointer as nextLayer will be null
		
	}

	@Override
	public void updateWeights(List<Double> layer_inputs, Layer nextLayer) {
		// No waits to update
		// Just remember the next layer so we can access it in the getDeltaw method
		this.nextLayer = nextLayer;
	}

	@Override
	public double getDeltaW(int unit_index) {
		// We only do backpropagation through the unit if it is the max
		// Return 0 if the unit is not the max for all possible window positions
		// Otherwise return the delta.w from the next layer
		
		// First convert unit_index to 2D coordinates
		int x = (unit_index/inputDimensions) / inputImageSize;
		int y = (unit_index/inputDimensions) % inputImageSize;
		int plate = unit_index%inputDimensions;
		double deltaw = 0;
		// So this pixel influences the output of units from 
		// (x-window + 1,y-window + 1) to (x, y) for each plate

		for (int i = x - windowSize + 1; i <= x; i++) {
			if (!(i >=0 && i <outputSideDim))
				continue;
			for (int j = y - windowSize + 1; j <= y; j++) {
				if (!(j>=0 && j <outputSideDim))
					continue;
				if (Math.abs(output.get((i*outputSideDim + j)*noPlates + plate) - inputValues.get(unit_index)) < 0.0001)
					deltaw += nextLayer.getDeltaW((i*outputSideDim + j)*noPlates + plate);
			}
		}

		return deltaw;
		
	}
	
}

class ANN {
	int[] layerSizes;
	ArrayList<Layer> layers;
	double eta;
	double alpha;
	public int noEpochs;
	public static int MAX_PATIENCE = 20;
	public static int SEED = 10;
	
	Vector<Vector<Double>> train;
	Vector<Vector<Double>> tune;
	Vector<Vector<Double>> test;
	
	public ANN(Vector<Vector<Double>> train) {
		this.layers = new ArrayList<Layer>();
		this.train = train;
	}
	
	public ANN(double eta, double alpha, int noEpochs, int[] layer_sizes, double[] dropouts, int input_size, Vector<Vector<Double>> train, 
			Vector<Vector<Double>> tune, Vector<Vector<Double>> test) {
		this.eta = eta;
		this.alpha = alpha;
		this.noEpochs = noEpochs;
		this.train = train;
		this.tune = tune;
		this.test = test;
		this.layerSizes = layer_sizes;
		this.layers = new ArrayList<Layer>();
		
		for (int i = 0; i < layer_sizes.length; i++) {
			if (i == 0)
				layers.add(new DenseLayer(layer_sizes[i], train.get(0).size(), eta, alpha, dropouts[i]));
			else
				layers.add(new DenseLayer(layer_sizes[i], layers.get(i-1).output_size, eta, alpha, dropouts[i]));
		}
	}
	
	public void add(Layer layer) {
		layers.add(layer);
	}
	
	public void trainANN() {
		// for each example train the ANN
		//Collections.shuffle(train);
		for (int i = 0; i < train.size(); i++) {
			// Now traverse through each layer and generate outputs
			for (int j = 0; j < layers.size(); j++) {
				if (j == 0) {
					layers.get(j).updateOutput(train.get(i));
				} else {
					layers.get(j).updateOutput(layers.get(j-1).output);
				}
			}
			
			// Now do backpropogation
			for (int j = layers.size()-1; j >= 0; j--) {
				if (j == layers.size()-1) {
					
					layers.get(j).updateWeights(train.get(i).get(train.get(i).size() - 1), layers.get(j-1).output);
				} else if (j == 0) {
					layers.get(j).updateWeights(train.get(i), layers.get(j+1));
				} else {
					layers.get(j).updateWeights(layers.get(j-1).output, layers.get(j+1));
				}
			}
			
		}
	}
	
	public double getAccuracy(Vector<Vector<Double>> samples){
		double acc = 0;
		for(Vector<Double> sample : samples) {
			ArrayList<Double> y_label = this.getLabel(sample);
			// We use Least Squares Loss for comparing the 
			// label outputted by the ANN and the sample's label
			int y_ind = y_label.indexOf(Collections.max(y_label));
			int ind = sample.get(sample.size() - 1).intValue();
			if (ind == y_ind) {
				acc++;
			}
		}
		return (double)acc/samples.size();
	}
	
	public ArrayList<Double> getLabel(Vector<Double> inst) {
		for (int j = 0; j < layers.size(); j++) {
			double back_dropout = layers.get(j).dropout;
			layers.get(j).dropout = 0;
			if (j == 0) {
				layers.get(j).updateOutput(inst);
			} else {
				layers.get(j).updateOutput(layers.get(j-1).output);
			}
			layers.get(j).dropout = back_dropout;
		}
		return layers.get(layers.size() - 1).output;
	}
	
	@Override
	public Object clone() throws CloneNotSupportedException{
		return super.clone();
	}
}

public class Lab3 {
    
	private static int     imageSize = 32; // Images are imageSize x imageSize.  The provided data is 128x128, but this can be resized by setting this value (or passing in an argument).  
	                                       // You might want to resize to 8x8, 16x16, 32x32, or 64x64; this can reduce your network size and speed up debugging runs.
	                                       // ALL IMAGES IN A TRAINING RUN SHOULD BE THE *SAME* SIZE.
	private static enum    Category { airplanes, butterfly, flower, grand_piano, starfish, watch };  // We'll hardwire these in, but more robust code would not do so.
	
	private static final Boolean    useRGB = true; // If true, FOUR units are used per pixel: red, green, blue, and grey.  If false, only ONE (the grey-scale value).
	public static       int unitsPerPixel = (useRGB ? 4 : 1); // If using RGB, use red+blue+green+grey.  Otherwise just use the grey value.
			
	private static String    modelToUse = "oneLayer"; // Should be one of { "perceptrons", "oneLayer", "deep" };  You might want to use this if you are trying approaches other than a Deep ANN.
	private static int       inputVectorSize;         // The provided code uses a 1D vector of input features.  You might want to create a 2D version for your Deep ANN code.  
	                                                  // Or use the get2DfeatureValue() 'accessor function' that maps 2D coordinates into the 1D vector.  
	                                                  // The last element in this vector holds the 'teacher-provided' label of the example.

	private static double eta       =    0.01, fractionOfTrainingToUse = 1.00, dropoutRate = 0.00; // To turn off drop out, set dropoutRate to 0.0 (or a neg number).
	private static int    maxEpochs = 1000; // Feel free to set to a different value.

	public static void main(String[] args) {
		String trainDirectory = "trainset/";
		String  tuneDirectory = "tuneset/";
		String  testDirectory = "testset/";
		
        if(args.length > 5) {
            System.err.println("Usage error: java Lab3 <train_set_folder_path> <tune_set_folder_path> <test_set_foler_path> <imageSize>");
            System.exit(1);
        }
        if (args.length >= 1) { trainDirectory = args[0]; }
        if (args.length >= 2) {  tuneDirectory = args[1]; }
        if (args.length >= 3) {  testDirectory = args[2]; }
        if (args.length >= 4) {  imageSize     = Integer.parseInt(args[3]); }
    
		// Here are statements with the absolute path to open images folder
        File trainsetDir = new File(trainDirectory);
        File tunesetDir  = new File( tuneDirectory);
        File testsetDir  = new File( testDirectory);
        
        // create three datasets
		Dataset trainset = new Dataset();
        Dataset  tuneset = new Dataset();
        Dataset  testset = new Dataset();
        
        // Load in images into datasets.
        long start = System.currentTimeMillis();
        loadDataset(trainset, trainsetDir);
        System.out.println("The trainset contains " + comma(trainset.getSize()) + " examples.  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");
        
        start = System.currentTimeMillis();
        loadDataset(tuneset, tunesetDir);
        System.out.println("The  testset contains " + comma( tuneset.getSize()) + " examples.  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");
        
        start = System.currentTimeMillis();
        loadDataset(testset, testsetDir);
        System.out.println("The  tuneset contains " + comma( testset.getSize()) + " examples.  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");
        
        
        // Now train a Deep ANN.  You might wish to first use your Lab 2 code here and see how one layer of HUs does.  Maybe even try your perceptron code.
        // We are providing code that converts images to feature vectors.  Feel free to discard or modify.
        start = System.currentTimeMillis();
        trainANN(trainset, tuneset, testset);
        System.out.println("\nTook " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + " to train.");
        
    }

	public static void loadDataset(Dataset dataset, File dir) {
        for(File file : dir.listFiles()) {
            // check all files
             if(!file.isFile() || !file.getName().endsWith(".jpg")) {
                continue;
            }
            //String path = file.getAbsolutePath();
            BufferedImage img = null, scaledBI = null;
            try {
                // load in all images
                img = ImageIO.read(file);
                // every image's name is in such format:
                // label_image_XXXX(4 digits) though this code could handle more than 4 digits.
                String name = file.getName();
                int locationOfUnderscoreImage = name.indexOf("_image");
                
                // Resize the image if requested.  Any resizing allowed, but should really be one of 8x8, 16x16, 32x32, or 64x64 (original data is 128x128).
                if (imageSize != 128) {
                    scaledBI = new BufferedImage(imageSize, imageSize, BufferedImage.TYPE_INT_RGB);
                    Graphics2D g = scaledBI.createGraphics();
                    g.drawImage(img, 0, 0, imageSize, imageSize, null);
                    g.dispose();
                }
                
                Instance instance = new Instance(scaledBI == null ? img : scaledBI, name.substring(0, locationOfUnderscoreImage));

                dataset.add(instance);
            } catch (IOException e) {
                System.err.println("Error: cannot load in the image file");
                System.exit(1);
            }
        }
    }
	///////////////////////////////////////////////////////////////////////////////////////////////
	
	private static Category convertCategoryStringToEnum(String name) {
		if ("airplanes".equals(name))   return Category.airplanes; // Should have been the singular 'airplane' but we'll live with this minor error.
		if ("butterfly".equals(name))   return Category.butterfly;
		if ("flower".equals(name))      return Category.flower;
		if ("grand_piano".equals(name)) return Category.grand_piano;
		if ("starfish".equals(name))    return Category.starfish;
		if ("watch".equals(name))       return Category.watch;
		throw new Error("Unknown category: " + name);		
	}

	private static double getRandomWeight(int fanin, int fanout) { // This is one 'rule of thumb' for initializing weights.  Fine for perceptrons and one-layer ANN at least.
		double range = Math.max(Double.MIN_VALUE, 4.0 / Math.sqrt(6.0 * (fanin + fanout)));
		return (2.0 * random() - 1.0) * range;
	}
	
	// Map from 2D coordinates (in pixels) to the 1D fixed-length feature vector.
	private static double get2DfeatureValue(Vector<Double> ex, int x, int y, int offset) { // If only using GREY, then offset = 0;  Else offset = 0 for RED, 1 for GREEN, 2 for BLUE, and 3 for GREY.
		return ex.get(unitsPerPixel * (y * imageSize + x) + offset); // Jude: I have not used this, so might need debugging.
	}
	
	///////////////////////////////////////////////////////////////////////////////////////////////

    
	// Return the count of TESTSET errors for the chosen model.
    private static int trainANN(Dataset trainset, Dataset tuneset, Dataset testset) {
    	Instance sampleImage = trainset.getImages().get(0); // Assume there is at least one train image!
    	inputVectorSize = sampleImage.getWidth() * sampleImage.getHeight() * unitsPerPixel + 1; // The '-1' for the bias is not explicitly added to all examples (instead code should implicitly handle it).  The final 1 is for the CATEGORY.
    	
    	// For RGB, we use FOUR input units per pixel: red, green, blue, plus grey.  Otherwise we only use GREY scale.
    	// Pixel values are integers in [0,255], which we convert to a double in [0.0, 1.0].
    	// The last item in a feature vector is the CATEGORY, encoded as a double in 0 to the size on the Category enum.
    	// We do not explicitly store the '-1' that is used for the bias.  Instead code (to be written) will need to implicitly handle that extra feature.
    	System.out.println("\nThe input vector size is " + comma(inputVectorSize - 1) + ".\n");
    	
    	Vector<Vector<Double>> trainFeatureVectors = new Vector<Vector<Double>>(trainset.getSize());
    	Vector<Vector<Double>>  tuneFeatureVectors = new Vector<Vector<Double>>( tuneset.getSize());
    	Vector<Vector<Double>>  testFeatureVectors = new Vector<Vector<Double>>( testset.getSize());
		
        long start = System.currentTimeMillis();
		fillFeatureVectors(trainFeatureVectors, trainset);
        System.out.println("Converted " + trainFeatureVectors.size() + " TRAIN examples to feature vectors. Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");
        
        start = System.currentTimeMillis();
        fillFeatureVectors( tuneFeatureVectors,  tuneset);
        System.out.println("Converted " +  tuneFeatureVectors.size() + " TUNE  examples to feature vectors. Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");
        
        start = System.currentTimeMillis();
		fillFeatureVectors( testFeatureVectors,  testset);
        System.out.println("Converted " +  testFeatureVectors.size() + " TEST  examples to feature vectors. Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");
        
        System.out.println("\nTime to start learning!");
        
        // Call your Deep ANN here.  We recommend you create a separate class file for that during testing and debugging, but before submitting your code cut-and-paste that code here.
		
        if      ("perceptrons".equals(modelToUse)) return trainPerceptrons(trainFeatureVectors, tuneFeatureVectors, testFeatureVectors); // This is optional.  Either comment out this line or just right a 'dummy' function.
        else if ("oneLayer".equals(   modelToUse)) return trainOneHU(      trainFeatureVectors, tuneFeatureVectors, testFeatureVectors); // This is optional.  Ditto.
        else if ("deep".equals(       modelToUse)) return trainDeep(       trainFeatureVectors, tuneFeatureVectors, testFeatureVectors);
        return -1;
	}
    
	private static void fillFeatureVectors(Vector<Vector<Double>> featureVectors, Dataset dataset) {
		for (Instance image : dataset.getImages()) {
			featureVectors.addElement(convertToFeatureVector(image));
		}
	}

	private static Vector<Double> convertToFeatureVector(Instance image) {
		Vector<Double> result = new Vector<Double>(inputVectorSize);		

		for (int index = 0; index < inputVectorSize - 1; index++) { // Need to subtract 1 since the last item is the CATEGORY.
			if (useRGB) {
				int xValue = (index / unitsPerPixel) % image.getWidth(); 
				int yValue = (index / unitsPerPixel) / image.getWidth();
			//	System.out.println("  xValue = " + xValue + " and yValue = " + yValue + " for index = " + index);
				if      (index % 3 == 0) result.add(image.getRedChannel()  [xValue][yValue] / 255.0); // If unitsPerPixel > 4, this if-then-elseif needs to be edited!
				else if (index % 3 == 1) result.add(image.getGreenChannel()[xValue][yValue] / 255.0);
				else if (index % 3 == 2) result.add(image.getBlueChannel() [xValue][yValue] / 255.0);
				else                     result.add(image.getGrayImage()   [xValue][yValue] / 255.0); // Seems reasonable to also provide the GREY value.
			} else {
				int xValue = index % image.getWidth();
				int yValue = index / image.getWidth();
				result.add(                         image.getGrayImage()   [xValue][yValue] / 255.0);
			}
		}
		result.add((double) convertCategoryStringToEnum(image.getLabel()).ordinal()); // The last item is the CATEGORY, representing as an integer starting at 0 (and that int is then coerced to double).
		
		return result;
	}
	
	////////////////////  Some utility methods (cut-and-pasted from JWS' Utils.java file). ///////////////////////////////////////////////////
	
	private static final long millisecInMinute = 60000;
	private static final long millisecInHour   = 60 * millisecInMinute;
	private static final long millisecInDay    = 24 * millisecInHour;
	public static String convertMillisecondsToTimeSpan(long millisec) {
		return convertMillisecondsToTimeSpan(millisec, 0);
	}
	public static String convertMillisecondsToTimeSpan(long millisec, int digits) {
		if (millisec ==    0) { return "0 seconds"; } // Handle these cases this way rather than saying "0 milliseconds."
		if (millisec <  1000) { return comma(millisec) + " milliseconds"; } // Or just comment out these two lines?
		if (millisec > millisecInDay)    { return comma(millisec / millisecInDay)    + " days and "    + convertMillisecondsToTimeSpan(millisec % millisecInDay,    digits); }
		if (millisec > millisecInHour)   { return comma(millisec / millisecInHour)   + " hours and "   + convertMillisecondsToTimeSpan(millisec % millisecInHour,   digits); }
		if (millisec > millisecInMinute) { return comma(millisec / millisecInMinute) + " minutes and " + convertMillisecondsToTimeSpan(millisec % millisecInMinute, digits); }
		
		return truncate(millisec / 1000.0, digits) + " seconds"; 
	}

    public static String comma(int value) { // Always use separators (e.g., "100,000").
    	return String.format("%,d", value);    	
    }    
    public static String comma(long value) { // Always use separators (e.g., "100,000").
    	return String.format("%,d", value);    	
    }   
    public static String comma(double value) { // Always use separators (e.g., "100,000").
    	return String.format("%,f", value);    	
    }
    public static String padLeft(String value, int width) {
    	String spec = "%" + width + "s";
    	return String.format(spec, value);    	
    }
    
    /**
     * Format the given floating point number by truncating it to the specified
     * number of decimal places.
     * 
     * @param d
     *            A number.
     * @param decimals
     *            How many decimal places the number should have when displayed.
     * @return A string containing the given number formatted to the specified
     *         number of decimal places.
     */
    public static String truncate(double d, int decimals) {
    	double abs = Math.abs(d);
    	if (abs > 1e13)             { 
    		return String.format("%."  + (decimals + 4) + "g", d);
    	} else if (abs > 0 && abs < Math.pow(10, -decimals))  { 
    		return String.format("%."  +  decimals      + "g", d);
    	}
        return     String.format("%,." +  decimals      + "f", d);
    }
    
    /** Randomly permute vector in place.
     *
     * @param <T>  Type of vector to permute.
     * @param vector Vector to permute in place. 
     */
    public static <T> void permute(Vector<T> vector) {
    	if (vector != null) { // NOTE from JWS (2/2/12): not sure this is an unbiased permute; I prefer (1) assigning random number to each element, (2) sorting, (3) removing random numbers.
    		// But also see "http://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle" which justifies this.
    		/*	To shuffle an array a of n elements (indices 0..n-1):
 									for i from n - 1 downto 1 do
      								j <- random integer with 0 <= j <= i
      								exchange a[j] and a[i]
    		 */

    		for (int i = vector.size() - 1; i >= 1; i--) {  // Note from JWS (2/2/12): to match the above I reversed the FOR loop that Trevor wrote, though I don't think it matters.
    			int j = random0toNminus1(i + 1);
    			if (j != i) {
    				T swap =    vector.get(i);
    				vector.set(i, vector.get(j));
    				vector.set(j, swap);
    			}
    		}
    	}
    }
    
    public static Random randomInstance = new Random(638 * 838);  // Change the 638 * 838 to get a different sequence of random numbers.
    
    /**
     * @return The next random double.
     */
    public static double random() {
        return randomInstance.nextDouble();
    }

    /**
     * @param lower
     *            The lower end of the interval.
     * @param upper
     *            The upper end of the interval. It is not possible for the
     *            returned random number to equal this number.
     * @return Returns a random integer in the given interval [lower, upper).
     */
    public static int randomInInterval(int lower, int upper) {
    	return lower + (int) Math.floor(random() * (upper - lower));
    }


    /**
     * @param upper
     *            The upper bound on the interval.
     * @return A random number in the interval [0, upper).
     * @see Utils#randomInInterval(int, int)
     */
    public static int random0toNminus1(int upper) {
    	return randomInInterval(0, upper);
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////////////  Write your own code below here.  Feel free to use or discard what is provided.
    	
	private static int trainPerceptrons(Vector<Vector<Double>> trainFeatureVectors, Vector<Vector<Double>> tuneFeatureVectors, Vector<Vector<Double>> testFeatureVectors) {
		Vector<Vector<Double>> perceptrons = new Vector<Vector<Double>>(Category.values().length);  // One perceptron per category.

		for (int i = 0; i < Category.values().length; i++) {
			Vector<Double> perceptron = new Vector<Double>(inputVectorSize);  // Note: inputVectorSize includes the OUTPUT CATEGORY as the LAST element.  That element in the perceptron will be the BIAS.
			perceptrons.add(perceptron);
			for (int indexWgt = 0; indexWgt < inputVectorSize; indexWgt++) perceptron.add(getRandomWeight(inputVectorSize, 1)); // Initialize weights.
		}

		if (fractionOfTrainingToUse < 1.0) {  // Randomize list, then get the first N of them.
			int numberToKeep = (int) (fractionOfTrainingToUse * trainFeatureVectors.size());
			Vector<Vector<Double>> trainFeatureVectors_temp = new Vector<Vector<Double>>(numberToKeep);

			permute(trainFeatureVectors); // Note: this is an IN-PLACE permute, but that is OK.
			for (int i = 0; i <numberToKeep; i++) {
				trainFeatureVectors_temp.add(trainFeatureVectors.get(i));
			}
			trainFeatureVectors = trainFeatureVectors_temp;
		}
		
        int trainSetErrors = Integer.MAX_VALUE, tuneSetErrors = Integer.MAX_VALUE, best_tuneSetErrors = Integer.MAX_VALUE, testSetErrors = Integer.MAX_VALUE, best_epoch = -1, testSetErrorsAtBestTune = Integer.MAX_VALUE;
        long  overallStart = System.currentTimeMillis(), start = overallStart;
		
		for (int epoch = 1; epoch <= maxEpochs /* && trainSetErrors > 0 */; epoch++) { // Might still want to train after trainset error = 0 since we want to get all predictions on the 'right side of zero' (whereas errors defined wrt HIGHEST output).
			permute(trainFeatureVectors); // Note: this is an IN-PLACE permute, but that is OK.

            // CODE NEEDED HERE!
			
	        System.out.println("Done with Epoch # " + comma(epoch) + ".  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + " (" + convertMillisecondsToTimeSpan(System.currentTimeMillis() - overallStart) + " overall).");
	        reportPerceptronConfig(); // Print out some info after epoch, so you can see what experiment is running in a given console.
	        start = System.currentTimeMillis();
		}
    	System.out.println("\n***** Best tuneset errors = " + comma(best_tuneSetErrors) + " of " + comma(tuneFeatureVectors.size()) + " (" + truncate((100.0 *      best_tuneSetErrors) / tuneFeatureVectors.size(), 2) + "%) at epoch = " + comma(best_epoch) 
    						+ " (testset errors = "    + comma(testSetErrorsAtBestTune) + " of " + comma(testFeatureVectors.size()) + ", " + truncate((100.0 * testSetErrorsAtBestTune) / testFeatureVectors.size(), 2) + "%).\n");
    	return testSetErrorsAtBestTune;
	}
	
	private static void reportPerceptronConfig() {
		System.out.println(  "***** PERCEPTRON: UseRGB = " + useRGB + ", imageSize = " + imageSize + "x" + imageSize + ", fraction of training examples used = " + truncate(fractionOfTrainingToUse, 2) + ", eta = " + truncate(eta, 2) + ", dropout rate = " + truncate(dropoutRate, 2)	);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////   ONE HIDDEN LAYER

	private static boolean debugOneLayer               = false;  // If set true, more things checked and/or printed (which does slow down the code).
	private static int    numberOfHiddenUnits          = 50;
	
	private static int trainOneHU(Vector<Vector<Double>> trainFeatureVectors, Vector<Vector<Double>> tuneFeatureVectors, Vector<Vector<Double>> testFeatureVectors) {
	    long overallStart   = System.currentTimeMillis(), start = overallStart;
        int  trainSetErrors = Integer.MAX_VALUE, tuneSetErrors = Integer.MAX_VALUE, best_tuneSetErrors = Integer.MAX_VALUE, testSetErrors = Integer.MAX_VALUE, best_epoch = -1, testSetErrorsAtBestTune = Integer.MAX_VALUE;
        //ANN ann = new ANN(eta, 0.0, 1000, new int[]{numberOfHiddenUnits, Category.values().length}, new double[] {dropoutRate, 0}, unitsPerPixel, trainFeatureVectors, tuneFeatureVectors, testFeatureVectors);
		ANN ann = new ANN(trainFeatureVectors);
		Layer conv1 = new ConvolutionLayer(20, imageSize, 5, unitsPerPixel, eta, 0.0, dropoutRate);
		ann.add(conv1);
		//ann.add(new DenseLayer(numberOfHiddenUnits, 20 * (imageSize - 5 + 1) * (imageSize - 5 + 1), eta, 0.0, dropoutRate));
		Layer mpl = new MaxPoolingLayer(20, imageSize - 5 + 1, 2, 2, eta, 0.0, dropoutRate);
		ann.add(mpl);
		Layer conv2 = new ConvolutionLayer(20, (int)Math.sqrt(mpl.output_size/20), 5, 20, eta, 0.0, dropoutRate);
		ann.add(conv2);
		Layer mpl2 = new MaxPoolingLayer(20, (int)Math.sqrt(conv2.output_size/20), 2, 2, eta, 0.0, dropoutRate);
		ann.add(mpl2);
		ann.add(new DenseLayer(numberOfHiddenUnits, mpl2.output_size, eta, 0.0, dropoutRate));
		//ann.add(new DenseLayer(numberOfHiddenUnits, trainFeatureVectors.get(0).size(), eta, 0.0, dropoutRate));
		ann.add(new DenseLayer(Category.values().length, numberOfHiddenUnits, eta, 0.0, dropoutRate));
        for (int epoch = 1; epoch <= maxEpochs /* && trainSetErrors > 0 */; epoch++) { // Might still want to train after trainset error = 0 since we want to get all predictions on the 'right side of zero' (whereas errors defined wrt HIGHEST output).
			permute(trainFeatureVectors); // Note: this is an IN-PLACE permute, but that is OK.

            ann.trainANN();
            double acc = ann.getAccuracy(testFeatureVectors);
            System.out.println(acc);
	        System.out.println("Done with Epoch # " + comma(epoch) + ".  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + " (" + convertMillisecondsToTimeSpan(System.currentTimeMillis() - overallStart) + " overall).");
	        reportOneLayerConfig(); // Print out some info after epoch, so you can see what experiment is running in a given console.
	        start = System.currentTimeMillis();
		}
		
		System.out.println("\n***** Best tuneset errors = " + comma(best_tuneSetErrors) + " of " + comma(tuneFeatureVectors.size()) + " (" + truncate((100.0 *      best_tuneSetErrors) / tuneFeatureVectors.size(), 2) + "%) at epoch = " + comma(best_epoch) 
		                    + " (testset errors = "    + comma(testSetErrorsAtBestTune) + " of " + comma(testFeatureVectors.size()) + ", " + truncate((100.0 * testSetErrorsAtBestTune) / testFeatureVectors.size(), 2) + "%).\n");
    	return testSetErrorsAtBestTune;
	}
	
	private static void reportOneLayerConfig() {
		System.out.println(  "***** ONE-LAYER: UseRGB = " + useRGB + ", imageSize = " + imageSize + "x" + imageSize + ", fraction of training examples used = " + truncate(fractionOfTrainingToUse, 2) 
		        + ", eta = " + truncate(eta, 2)   + ", dropout rate = "      + truncate(dropoutRate, 2) + ", number HUs = " + numberOfHiddenUnits
			//	+ ", activationFunctionForHUs = " + activationFunctionForHUs + ", activationFunctionForOutputs = " + activationFunctionForOutputs
			//	+ ", # forward props = " + comma(forwardPropCounter)
				);
	//	for (Category cat : Category.values()) {  // Report the output unit biases.
	//		int catIndex = cat.ordinal();
    //
	//		System.out.print("  bias(" + cat + ") = " + truncate(weightsToOutputUnits[numberOfHiddenUnits][catIndex], 6));
	//	}   System.out.println();
	}

	// private static long forwardPropCounter = 0;  // Count the number of forward propagations performed.
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////  DEEP ANN Code


	private static int trainDeep(Vector<Vector<Double>> trainFeatureVectors, Vector<Vector<Double>> tuneFeatureVectors,	Vector<Vector<Double>> testFeatureVectors) {
		// You need to implement this method!
		return -1;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////

}
