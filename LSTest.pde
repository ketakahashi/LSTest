import java.util.Arrays;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresBuilder;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresOptimizer;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresProblem;
import org.apache.commons.math3.fitting.leastsquares.LevenbergMarquardtOptimizer;
import org.apache.commons.math3.fitting.leastsquares.MultivariateJacobianFunction;
import org.apache.commons.math3.geometry.euclidean.twod.Vector2D;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.Pair;
import java.util.List;

List<PVector> observedPoints;

double a = -1, b = -1;
boolean calcDone = false;

void setup()
{
  size(400, 400);
  stroke(255);
  observedPoints = new ArrayList<PVector>();
}

void draw()
{
  background(0);
  for (PVector p : observedPoints) 
    ellipse(p.x, p.y, 20, 20);
  line(0, (float)b, width, (float)(a * 400 + b));
}

void mousePressed() 
{
  if (mouseButton == LEFT) {
    if (calcDone) {
      observedPoints.clear();
      calcDone = false;
      a = b = -1;
    }
    observedPoints.add(new PVector(mouseX, mouseY));
  } else
    calc();
}

void calc()
{
  MultivariateJacobianFunction distancesToCurrentCenter = new MultivariateJacobianFunction() {
    public Pair<RealVector, RealMatrix> value(final RealVector point) {

      a = point.getEntry(0);
      b = point.getEntry(1);

      RealVector value = new ArrayRealVector(observedPoints.size());
      RealMatrix jacobian = new Array2DRowRealMatrix(observedPoints.size(), 2);

      for (int i = 0; i < observedPoints.size(); ++i) {
        PVector o = observedPoints.get(i);
        double modelI = (o.y - (a * o.x + b)) * (o.y - (a * o.x + b));
        value.setEntry(i, modelI);
        jacobian.setEntry(i, 0, a * o.x * o.x + b * o.x - o.x * o.y);
        jacobian.setEntry(i, 1, a * o.x + b - o.y);
      }

      return new Pair<RealVector, RealMatrix>(value, jacobian);
    }
  };

  double[] prescribedDistances = new double[observedPoints.size()];
  Arrays.fill(prescribedDistances, 0);

  LeastSquaresProblem problem = new LeastSquaresBuilder()
    .start(new double[] { 0.0, 0.0 })
    .model(distancesToCurrentCenter).target(prescribedDistances)
    .lazyEvaluation(false).maxEvaluations(1000).maxIterations(1000)
    .build();
  LeastSquaresOptimizer.Optimum optimum = new LevenbergMarquardtOptimizer().optimize(problem);
  a = optimum.getPoint().getEntry(0);
  b = optimum.getPoint().getEntry(1);
  println("RMS: " + optimum.getRMS());
  println("evaluations: " + optimum.getEvaluations());
  println("iterations: " + optimum.getIterations());
  calcDone = true;
}