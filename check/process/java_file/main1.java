import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        String end = "end";
        String current = "";
        while (!current.equals(end)) {
            current = scanner.nextLine();
            if (current.length() % 3 == 0) {
                System.out.print(current + " ");
            }
        }
    }
}