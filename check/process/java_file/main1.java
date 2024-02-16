import java.util.Scanner;
public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        if (n > 9 || n < 1) {
            System.out.println("Input error.");
            return;
        }
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                System.out.printf("%dX%d=%d\t", i, j, i * j);
            }
            System.out.println();
        }
    }
}