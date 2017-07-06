<?php 
    // 获取上传的图片
    $pic = $_FILES['pic']['tmp_name'];
    $upload_ret = false;

    if($pic){
        // 上传的路径，建议写物理路径
        $uploadDir = 'static/upload'; 
        // 创建文件夹  
        if(!file_exists($uploadDir)){        
            mkdir($uploadDir, 0777);    
        }    
        // 用时间戳来保存图片，防止重复
        $targetFile = $uploadDir . '/' . time() . $_FILES['pic']['name'];    
        // 将临时文件 移动到我们指定的路径，返回上传结果
        $upload_ret = move_uploaded_file($pic, $targetFile) ? true : false;
    }

    return $upload_ret;
?>