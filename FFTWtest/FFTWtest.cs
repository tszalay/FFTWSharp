using System;
using System.Runtime.InteropServices;
using FFTWSharp;

namespace FFTWSharp_test
{
    public class FFTWtest
    {
        const int sampleSize = 16384;
        const int repeatPlan = 10000;
        const bool forgetWisdom = false;

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
        fftwf_plan mplan1, mplan2, mplan3, mplan4;
        fftwf_complexarray mfin, mfout;
        fftw_plan mplan5;
        fftw_complexarray mdin, mdout;

        // length of arrays
        int fftLength = 0;

        // Initializes FFTW and all arrays
        // n: Logical size of the transform
        public FFTWtest(int n)
        {
            Console.WriteLine($"Start testing with n = {n.ToString("#,0")} complex numbers. All plans will be executed {repeatPlan.ToString("#,0")} times on a single thread.");
            Console.WriteLine("Please wait, creating plans...");
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
            fplan2 = fftwf.dft_1d(n, hin.AddrOfPinnedObject(), hout.AddrOfPinnedObject(), fftw_direction.Forward, fftw_flags.Estimate);
            fplan3 = fftwf.dft_1d(n, hout.AddrOfPinnedObject(), pin, fftw_direction.Backward, fftw_flags.Measure);
            // end with transforming back to original array
            fplan4 = fftwf.dft_1d(n, hout.AddrOfPinnedObject(), hin.AddrOfPinnedObject(), fftw_direction.Backward, fftw_flags.Estimate);
            // and check a quick one with doubles, just to be sure
            fplan5 = fftw.dft_1d(n, hdin.AddrOfPinnedObject(), hdout.AddrOfPinnedObject(), fftw_direction.Backward, fftw_flags.Measure);

            // create a managed plan as well
            mfin = new fftwf_complexarray(fin);
            mfout = new fftwf_complexarray(fout);
            mdin = new fftw_complexarray(din);
            mdout = new fftw_complexarray(dout);

            mplan1 = fftwf_plan.dft_1d(n, mfin, mfout, fftw_direction.Forward, fftw_flags.Estimate);
            mplan2 = fftwf_plan.dft_1d(n, mfin, mfout, fftw_direction.Forward, fftw_flags.Measure);
            mplan3 = fftwf_plan.dft_1d(n, mfin, mfout, fftw_direction.Forward, fftw_flags.Patient);

            mplan4 = fftwf_plan.dft_1d(n, mfout, mfin, fftw_direction.Backward, fftw_flags.Measure);
            
            mplan5 = fftw_plan.dft_1d(n, mdin, mdout, fftw_direction.Forward, fftw_flags.Measure);


            // fill our arrays with an arbitrary complex sawtooth-like signal
            for (int i = 0; i < n * 2; i++) fin[i] = i % 50;
            for (int i = 0; i < n * 2; i++) fout[i] = i % 50;
            for (int i = 0; i < n * 2; i++) din[i] = i % 50;

            // copy managed arrays to unmanaged arrays
            Marshal.Copy(fin, 0, pin, n * 2);
            Marshal.Copy(fout, 0, pout, n * 2);

            Console.WriteLine();
        }

        public void TestAll()
        {
            TestPlan(fplan1, "single (malloc) | pin,  pout,  Forward,  Estimate");
            TestPlan(fplan2, "single (pinned) | hin,  hout,  Forward,  Estimate");
            TestPlan(fplan3, "single (pinned) | hout, pin,   Backward, Measure ");

            // set fin to 0, and try to refill it from a backwards fft from fout (aka hin/hout)
            for (int i = 0; i < fftLength * 2; i++) fin[i] = 0;

            TestPlan(fplan4, "single (pinned) | hout, hin,   Backward, Estimate");

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

            TestPlan(fplan5, "double (pinned) | hdin, hdout, Backward, Measure ");


            Console.WriteLine();
            TestPlan(mplan1, "#1 single (managed) | mfin, mfout, Forward,  Estimate");
            TestPlan(mplan2, "#2 single (managed) | mfin, mfout, Forward,  Measure ");
            TestPlan(mplan3, "#3 single (managed) | mfin, mfout, Forward,  Patient ");
            Console.WriteLine();

            // fill our input array with an arbitrary complex sawtooth-like signal
            for (int i = 0; i < fftLength * 2; i++) fin[i] = i % 50;
            for (int i = 0; i < fftLength * 2; i++) fout[i] = 0;

            mfin.SetData(fin);
            mfout.SetData(fout);
            TestPlan(mplan2, "#2 single (managed) | mfin, mfout, Forward,  Measure ");

            fout = mfout.GetData_Float();  // let's see what's in mfout
                                           // at this point mfout contains the FFT'd mfin

            TestPlan(mplan4, "#4 single (managed) | mfout, mfin, Backward, Measure ");
            // at this point we have transfarred backwards the mfout into mfin, so mfin should be very close to the original complex sawtooth-like signal

            fin = mfin.GetData_Float();
            for (int i = 0; i < fftLength * 2; i++) fin[i] /= fftLength;
            // at this point fin should be very close to the original (sawtooth-like) signal

            // check and see how we did, don't say anyt
            for (int i = 0; i < fftLength * 2; i++)
            {
                // check against original values
                if (System.Math.Abs(fin[i] - (i % 50)) > 1e-3)
                {
                    System.Console.WriteLine("FFTW consistency error!");
                    return;
                }
            }

            Console.WriteLine();

            TestPlan(mplan2, "#2 single (managed) | mfin, mfout, Forward,  Measure ");
            TestPlan(mplan5, "#5 double (managed) | mdin, mdout, Forward,  Measure ");
            Console.WriteLine();
        }

        // Tests a single plan, displaying results
        // plan: Pointer to plan to test
        public void TestPlan(object plan, string planName)
        {
            // a: adds, b: muls, c: fmas
            double a = 0, b = 0, c = 0;

            int start = System.Environment.TickCount;

            if (plan is IntPtr)
            {
                IntPtr umplan = (IntPtr)plan;

                for (int i = 0; i < repeatPlan; i++)
                {
                        fftwf.execute(umplan);
                }

                fftwf.flops(umplan, ref a, ref b, ref c);
            }

            if (plan is fftw_plan)
            {
                fftw_plan mplan = (fftw_plan)plan;

                for (int i = 0; i < repeatPlan; i++)
                {
                    mplan.Execute();
                }

                fftw.flops(mplan.Handle, ref a, ref b, ref c);
            }

            if (plan is fftwf_plan)
            {
                fftwf_plan mplan = (fftwf_plan)plan;

                for (int i = 0; i < repeatPlan; i++)
                {
                    mplan.Execute();
                }

                fftwf.flops(mplan.Handle, ref a, ref b, ref c);
            }

            double mflops = (((a + b + 2 * c)) * repeatPlan) / (1024 * 1024);
            long ticks = (System.Environment.TickCount - start);

            //Console.WriteLine($"Plan '{planName}': {ticks.ToString("#,0")} us | mflops: {FormatNumber(mflops)} | mflops/s: {(1000*mflops/ticks).ToString("#,0.0")}");
            Console.WriteLine("Plan '{0}': {1,8:N0} us | mflops: {2,8:N0} | mflops/s: {3,8:N0}", planName, ticks, mflops, (1000 * mflops / ticks));
        }

        // Releases all memory used by FFTW/C#
        ~FFTWtest()
        {
            fftwf.export_wisdom_to_filename("wisdom.wsd");

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
            if (forgetWisdom)
            {
                fftwf.fftwf_forget_wisdom();
            }
            else
            {
                Console.WriteLine("Importing wisdom (wisdom speeds up the plan creation process, if that plan was previously created at least once)");
                fftwf.import_wisdom_from_filename("wisdom.wsd");
            }

            // initialize our FFTW test class
            FFTWtest test = new FFTWtest(sampleSize); 

            // run the tests, print debug output
            test.TestAll();

            // pause for user input, then quit
            System.Console.WriteLine("\nDone. Press any key to exit.");
            String str = System.Console.ReadLine();
        }
    }
}