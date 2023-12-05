import java.util.Scanner;

public class Main{
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        int num = sc.nextInt();
        if(num<=0){
            System.out.println("error");
        }else if(Tool.isPower(num)){
            System.out.println("yes");
        }else{
            System.out.println("no");
        }
    }
}

class Tool{
    public static boolean isPower(int num){
        if(num%2==0){
            return isPower(num/2);
        }
        return true;
    }
}