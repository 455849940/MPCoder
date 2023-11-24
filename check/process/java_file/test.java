import java.util.*;
import java.lang.*;

// 主函数
public class test {
    public static void main(String[] args) {
        MyClass obj = new MyClass();  // 创建MyClass的对象
        obj.myFunction();  // 调用MyClass中的函数
        System.out.println("Hello from MyClass!");
    }
}
class MyClass {
    public void myFunction() {
        System.out.println("Hello from MyClass!");
    }
}
