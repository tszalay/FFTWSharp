using System;
using System.Runtime.InteropServices;
using FFTWSharp;

namespace FFTWSharp_test
{
    public class FFTWtest
    {
        //pointers to unmanaged arrays
        IntPtr pin, pout;

        //managed arrays
        float[] fin, fout;
        double[] din, dout;

        //handles to managed arrays, keeps them pinned in memory
        GCHandle hin, hout, hdin, hdout;

        //pointers to the FFTW plan objects
        IntPtr fplan1, fplan2, fplan3, fplan4, fplan5;

        // and an example of the managed interface
        fftw_plan mplan;
        fftw_complexarray min, mout;

        // length of arrays
        int fftLength = 0;

        // Initializes FFTW and all arrays
        // n: Logical size of the transform
        public FFTWtest(int n)
        {
            System.Console.WriteLine("Starting test with n = " + n + " complex numbers");
            fftLength = n;

            // create two unmanaged arrays, properly aligned
            pin = fftwf.malloc(n * 8);
            pout = fftwf.malloc(n * 8);

            // create two managed arrays, possibly misalinged
            // n*2 because we are dealing with complex numbers
            fin = new float[n * 2];
            fout = new float[n * 2];
            // and two more for double FFTW
            din = new double[n * 2];
            dout = new double[n * 2];

            // get handles and pin arrays so the GC doesn't move them
            hin = GCHandle.Alloc(fin, GCHandleType.Pinned);
            hout = GCHandle.Alloc(fout, GCHandleType.Pinned);
            hdin = GCHandle.Alloc(din, GCHandleType.Pinned);
            hdout = GCHandle.Alloc(dout, GCHandleType.Pinned);

            // create a few test transforms
            fplan1 = fftwf.dft_1d(n, pin, pout, fftw_direction.Forward, fftw_flags.Estimate);
            fplan2 = fftwf.dft_1d(n, hin.AddrOfPinnedObject(), hout.AddrOfPinnedObject(),
                fftw_direction.Forward, fftw_flags.Estimate);
            fplan3 = fftwf.dft_1d(n, hout.AddrOfPinnedObject(), pin,
                fftw_direction.Backward, fftw_flags.Measure);
            // end with transforming back to original array
            fplan4 = fftwf.dft_1d(n, hout.AddrOfPinnedObject(), hin.AddrOfPinnedObject(),
                fftw_direction.Backward, fftw_flags.Estimate);
            // and check a quick one with doubles, just to be sure
            fplan5 = fftw.dft_1d(n, hdin.AddrOfPinnedObject(), hdout.AddrOfPinnedObject(),
                fftw_direction.Backward, fftw_flags.Measure);

            // create a managed plan as well
            min = new fftw_complexarray(din);
            mout = new fftw_complexarray(dout);
            mplan = fftw_plan.dft_1d(n, min, mout, fftw_direction.Forward, fftw_flags.Estimate);

            // fill our arrays with an arbitrary complex sawtooth-like signal
            for (int i = 0; i < n * 2; i++)
                fin[i] = i % 50;
            for (int i = 0; i < n * 2; i++)
                fout[i] = i % 50;
            for (int i = 0; i < n * 2; i++)
                din[i] = i % 50;

            // copy managed arrays to unmanaged arrays
            Marshal.Copy(fin, 0, pin, n * 2);
            Marshal.Copy(fout, 0, pout, n * 2);
        }

        public void TestAll()
        {
            System.Console.WriteLine("Testing single precision:\n");
            TestPlan(fplan1);
            TestPlan(fplan2);
            TestPlan(fplan3);
            // set fin to 0, and try to refill it from a backwards fft from fout (aka hin/hout)
            for (int i = 0; i < fftLength * 2; i++)
                fin[i] = 0;

            TestPlan(fplan4);

            // check and see how we did, don't say anyt
            for (int i = 0; i < fftLength * 2; i++)
            {
                // check against original values
                // note that we need to scale down by length, due to FFTW scaling by N
                if (System.Math.Abs(fin[i]/fftLength - (i % 50)) > 1e-3)
                {
                    System.Console.WriteLine("FFTW consistency error!");
                    return;
                }
            }

            System.Console.WriteLine("FFT consistency check ok.\n\nTesting double precision:\n");

            TestPlan(fplan5);

            System.Console.WriteLine("Testing managed interface:\n");

            mplan.Execute();

            // yeah alright so this was kind of a trivial test and of course it's gonna work. but still.
            System.Console.WriteLine("Ok.");
        }

        // Tests a single plan, displaying results
        // plan: Pointer to plan to test
        public void TestPlan(IntPtr plan)
        {
            int start = System.Environment.TickCount;
            for (int i = 0; i < 1000; i++)
                fftwf.execute(plan);
            Console.WriteLine("Time per plan: {0} us",
                (System.Environment.TickCount - start));
            // a: adds, b: muls, c: fmas
            double a = 0, b = 0, c = 0;
            fftwf.flops(plan, ref a, ref b, ref c);
            Console.WriteLine("Approx. flops: {0}\n", (a + b + 2 * c));
        }

        // Releases all memory used by FFTW/C#
        ~FFTWtest()
        {
            // it is essential that you call these after finishing
            // that's why they're in the destructor. See Also: RAII
            fftwf.free(pin);
            fftwf.free(pout);
            fftwf.destroy_plan(fplan1);
            fftwf.destroy_plan(fplan2);
            fftwf.destroy_plan(fplan3);
            fftwf.destroy_plan(fplan4);
            fftwf.destroy_plan(fplan5);
            hin.Free();
            hout.Free();
        }

        static void Main(string[] args)
        {
            // initialize our FFTW test class
            FFTWtest test = new FFTWtest(16384);

            // run the tests, print debug output
            test.TestAll();

            // pause for user input, then quit
            System.Console.WriteLine("\nDone. Press any key to exit.");
            String str = System.Console.ReadLine();
        }
    }
}