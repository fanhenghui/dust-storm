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
    }   
}