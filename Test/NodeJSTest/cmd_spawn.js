const fs = require('fs');
const child_process = require('child_process');

for(var i=0; i<3; i++) {
   var worker_process = child_process.spawn("run.bat");	
   worker_process.on('close', function (code) {
      console.log('子进程已退出，退出码 ' + code);
   });
   console.log("thread " , i);
}

// free = child_process.fork('run.bat', ['']);
// // 捕获标准输出并将其打印到控制台 
// free.stdout.on('data', function (data) { 
// console.log('standard output:\n' + data); 
// });
// // 捕获标准错误输出并将其打印到控制台 
// free.stderr.on('data', function (data) { 
// console.log('standard error output:\n' + data); 
// });
// // 注册子进程关闭事件 
// free.on('exit', function (code, signal) { 
// console.log('child process eixt ,exit:' + code); 
// });

console.log("process quit");