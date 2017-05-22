function change_image()
{	
	//window.alert("welcome");
	console.log("welcome");
	element = document.getElementById("my_img");
	if(element.src.match("./images/lung.jpg"))
	{
		element.src = "./images/lung2.png";
	}
	else
	{
		element.src = "./images/lung.jpg";
	}
}
"use strict";

var txt="";
function show_date()
{
	document.write(Date());
	//debugger;
	try
	{
		//addddinet("");
	}
	catch(err)
	{
		txt = "Error \n";
		txt += err.message +"\n";
		alert(txt);
	}
	
}

function add()
{
	var paras = document.getElementsByName("para");
	var result = 0;
	for(var i = 0; i<paras.length ; ++i)
	{
		result += Number(paras[i].value);
	}
	console.log(result);
	ele = document.getElementById("add_result");
	ele.innerHTML = result;
}

function test()
{
	//var cars = new Array("Saab" , "Volvo" , "BMW");
	//var cars = ["Saab" , "Volvo" , "BMW"];
	//document.write(cars[1]);
	
	//var person = {
	//	firstname:"Jonn" , lastname:"Doe" , id:1556};
	//document.write(person.firstname);
	
	var text = '{ "sites" : [' +
'{ "name":"Runoob" , "url":"www.runoob.com" },' +
'{ "name":"Google" , "url":"www.google.com" },' +
'{ "name":"Taobao" , "url":"www.taobao.com" } ]}';
	var obj = JSON.parse(text);
	var name = obj.sites[0].name;
	document.write(name);
}

function preview_image(imgFile)
{
    var filextension=imgFile.value.substring(imgFile.value.lastIndexOf("."),imgFile.value.length);
    filextension=filextension.toLowerCase();
    if ((filextension!='.jpg')&&(filextension!='.gif')&&(filextension!='.jpeg')&&(filextension!='.png')&&(filextension!='.bmp'))
    {
		alert("对不起，系统仅支持标准格式的照片，请您调整格式后重新上传，谢谢 !");
		imgFile.focus();
    }
    else
    {
		var path;
		if(document.all)//IE
		{
			imgFile.select();
			path = document.selection.createRange().text;
	
			document.getElementById("imgPreview").innerHTML="";
			document.getElementById("imgPreview").style.filter = "progid:DXImageTransform.Microsoft.AlphaImageLoader(enabled='true',sizingMethod='scale',src=\"" + path + "\")";//使用滤镜效果  
		}
		else//FF
		{
			path = window.URL.createObjectURL(imgFile.files[0]);
			//path = imgFile.files[0].getAsDataURL();
			document.getElementById("imgPreview").innerHTML = "<img id='img1' width='120px' height='100px' src='"+path+"'/>";
			// document.getElementById("img1").src = path;
		}
    }
}

function upload_image()
{
	var imgFile = document.getElementsByName("file");
	console.log(imgFile[0].value);
}