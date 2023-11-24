import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int a = sc.nextInt();
        int b = sc.nextInt();
        int[] c = new int[a];
        int[] d = new int[b];
        for (int i = 0; i < a; i++) {
            c[i] = sc.nextInt();
        }
        for (int i = 0; i < b; i++) {
            d[i] = sc.nextInt();
        }
        String winner = "A";
        if (a > b) {
            winner = "A";
        } else if (a < b) {
            winner = "B";
        } else {
            if (c[0] > d[0]) {
                winner = "A";
            } else if (c[0] < d[0]) {
                winner = "B";
            } else if (c[1] > d[1]) {
                winner = "A";
            } else if (c[1] < d[1]) {
                winner = "B";
            } else if (c[2] > d[2]) {
                winner = "A";
            } else if (c[2] < d[2]) {
                winner = "B";
            } else if (c[3] > d[3]) {
                winner = "A";
            } else if (c[3] < d[3]) {
                winner = "B";
            } else if (c[4] > d[4]) {
                winner = "A";
            } else if (c[4] < d[4]) {
                winner = "B";
            }
        }
        System.out.println(c[0] + " " + c[1] + " " + c[2]);
        System.out.println(winner);
    }
}