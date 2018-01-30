const crypto = require('crypto');
const uuid = require('uuid/v4');

module.exports = {
    md5(str) {
        let hash = crypto.createHash('md5');
        hash.update(str);
        return hash.digest('hex');
    },
    
    uuid() {
        return uuid().replace(/\-+/g, '');
    },
    
    getIPAddress(){  
        let interfaces = require('os').networkInterfaces();  
        for(let devName in interfaces){  
            let iface = interfaces[devName];  
              for(let i=0;i<iface.length;i++){  
                let alias = iface[i];  
                   if(alias.family === 'IPv4' && alias.address !== '127.0.0.1' && !alias.internal){  
                         return alias.address;  
                   }  
              }  
        }  
    } 

}