// SPDX-License-Identifier: GPL-3.0-only
// Drives the reference MicroRTS Java engine (JNIGridnetClientSelfPlay) from stdin
// so tools/parity/check_parity.py can compare it step by step with microrts_pz.
//
// Protocol (one line per message, space-separated ints):
//   in : "STEP n0 <n0*8 ints> n1 <n1*8 ints>"   (rows: cell, type, move, harvest, return, prod_dir, prod_type, attack)
//   in : "QUIT"
//   out: after start and after each STEP, one line per player p in {0,1}:
//        "P <p> <rewards...> | <obs ints...> | <mask ints...>" then "DONE <0|1>" then "RES <r0> <r1>"
import ai.reward.*;
import rts.PartiallyObservableGameState;
import rts.units.UnitTypeTable;
import tests.JNIGridnetClientSelfPlay;
import java.io.*;
import java.util.*;

public class ParityHarness {
    public static void main(String[] args) throws Exception {
        String mapPath = args[0];
        int version = Integer.parseInt(args[1]);
        partial = args.length > 2 && args[2].equals("1");
        UnitTypeTable utt = new UnitTypeTable(version);
        RewardFunctionInterface[] rfs = new RewardFunctionInterface[] {
            new WinLossRewardFunction(), new ResourceGatherRewardFunction(), new ProduceWorkerRewardFunction(),
            new ProduceBuildingRewardFunction(), new AttackRewardFunction(), new ProduceCombatUnitRewardFunction()};
        JNIGridnetClientSelfPlay c = new JNIGridnetClientSelfPlay(rfs, "", mapPath, utt, partial);
        c.reset();
        PrintStream out = new PrintStream(new BufferedOutputStream(System.out), false);
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
        dump(c, out, false);
        String line;
        while ((line = in.readLine()) != null) {
            StringTokenizer st = new StringTokenizer(line);
            String cmd = st.nextToken();
            if (cmd.equals("QUIT")) break;
            int[][][] acts = new int[2][][];
            for (int p = 0; p < 2; p++) {
                int n = Integer.parseInt(st.nextToken());
                acts[p] = new int[n][8];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < 8; j++) acts[p][i][j] = Integer.parseInt(st.nextToken());
            }
            c.gameStep(acts[0], acts[1]);
            dump(c, out, c.gs.gameover());
        }
        out.flush();
    }

    static boolean partial;

    static void dump(JNIGridnetClientSelfPlay c, PrintStream out, boolean done) throws Exception {
        for (int p = 0; p < 2; p++) {
            StringBuilder sb = new StringBuilder("P ").append(p);
            for (double r : c.getResponse(p).reward) sb.append(' ').append((int) r);
            sb.append(" |");
            // In partial-observability mode the Java client returns an observation built
            // from a PartiallyObservableGameState snapshotted *before* the tick (stale unit
            // list and actions). microrts_pz observes the post-tick state instead, so compare
            // against a fresh snapshot here.
            int[][][] obs = partial ? new PartiallyObservableGameState(c.gs, p).getVectorObservation(p)
                                    : c.getResponse(p).observation;
            for (int[][] plane : obs)
                for (int[] row : plane) for (int v : row) sb.append(' ').append(v);
            sb.append(" |");
            for (int[][] row : c.getMasks(p)) for (int[] cell : row) for (int v : cell) sb.append(' ').append(v);
            out.println(sb);
        }
        out.println("DONE " + (done ? 1 : 0));
        out.println("RES " + c.gs.getPlayer(0).getResources() + " " + c.gs.getPlayer(1).getResources());
        out.flush();
    }
}
