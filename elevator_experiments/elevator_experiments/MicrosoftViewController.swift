//
//  MicrosoftViewController.swift
//  elevator_experiments
//
//  Created by Pankaj Sharma on 1/30/23.
//

import SwiftUI
import MSAL
import Foundation


//let helper = Helpers()

//let kClientID = "3de9c462-dfd5-4082-a1af-b0e62822b8ab"
//let kRedirectUri = "msauth.com.microsoft.identitysample.MSALiOS://auth"
//let kAuthority = "https://login.microsoftonline.com/common"
//let kGraphEndpoint = "https://graph.microsoft.com/"

class MicrosoftViewController:UIViewController{
    override func viewDidLoad() {
        super.viewDidLoad()
    }
    
    var accessToken: String = ""
    private var application :Any
    private var accountIdentifier :String = ""
    @Binding var elProb: [String]

    init(value: Binding<[String]>) {
        do {
            self._elProb = value
            let authority = try MSALAuthority(url: URL(string: helper.msalConfig.authorityURL)!)
            
            let pcaConfig = MSALPublicClientApplicationConfig(clientId: helper.msalConfig.clientId, redirectUri: helper.msalConfig.redirectURI, authority: authority)
            application = try MSALPublicClientApplication(configuration: pcaConfig)
            super.init(nibName: nil, bundle: nil)
        }catch let error as NSError{
            application = ""
            super.init(nibName: nil, bundle: nil)
            print(error.localizedDescription)
        }

    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        let btn = UIButton(frame: CGRect(x:20, y: self.view.frame.height - 100, width: self.view.frame.width - 40, height: 52))
        btn.backgroundColor = .systemBlue
        btn.setTitle("Sign In", for: .normal)
        btn.setTitleColor(.white, for: .normal)
        btn.addTarget(self, action: #selector(buttonTapped), for: .touchUpInside)
        
        self.view.addSubview(btn)
    }
    
    @objc func buttonTapped(_ sender: UIButton) {
        /*do {*/
            var isSignIn = false
            if (sender.currentTitle == "Sign In"){
                isSignIn = true

            }else{
                isSignIn = false
            }
            
            if (!isSignIn){
                print("Sign out")
                Task {
                    let webViewParameters = MSALWebviewParameters(authPresentationViewController: self)
                    let signoutParameters = MSALSignoutParameters(webviewParameters: webViewParameters)
                    signoutParameters.signoutFromBrowser = false

                    
                    let account = try (self.application as! MSALPublicClientApplication).account(forIdentifier: self.accountIdentifier )
                    print(account)
                    try await (self.application as!
                         MSALPublicClientApplication).signout(with: account, signoutParameters: signoutParameters)
                    
                    sender.setTitle("Sign In", for: .normal)
                    sender.backgroundColor = .blue
                    
                    self.elProb[0]="0.00"
                    self.elProb[1]="0.00"
                    self.elProb[2]="0.00"
                    
                }

            }else{
                
                let webViewParameters = MSALWebviewParameters(authPresentationViewController: self)
                let interactiveParameters = MSALInteractiveTokenParameters(scopes: ["user.read","Files.ReadWrite.All"], webviewParameters: webViewParameters)
                interactiveParameters.promptType = .selectAccount
                (self.application as! MSALPublicClientApplication).acquireToken(with: interactiveParameters){ (result, error) in
                    guard let result = result else {
                        print(error!)
                        return
                    }
                    helper.accessToken = result.accessToken
                    print(result.tenantProfile.claims!["name"]!)
    //Start
                    
                    
                    /*
                    let graphURI = "https://graph.microsoft.com/v1.0/me/drive/root:/elevator_experiments.xlsx:/workbook/worksheets"
                    let url = URL(string: graphURI)
                    var request = URLRequest(url: url!)

                    // Set the Authorization header for the request. We use Bearer tokens, so we specify Bearer + the token we got from the result
                    request.setValue("Bearer \(result.accessToken)", forHTTPHeaderField: "Authorization")
                    
                    let task = URLSession.shared.dataTask(with: request) { data, response, error in
                        

                        if let error = error {
                            print("Couldn't get graph result: \(error)")
                            return
                        }
                        
                        if let result1 = try? JSONSerialization.jsonObject(with: data!, options: []) as? [String: Any]
                        {
                            
                            if let id = result1["value"] as? [String: Any]{
                                print("result = \(id)")
                            }else{
                                //print("Raw = \(result1)")
                                //print("Original = \(result1["value"])")
                                let json = result1["value"] as? [[String: Any]]

                                var id = String(describing: json?[0]["id"])
                                id = String(id.dropLast(2))
                                id = String(id.dropFirst(10))
                                print(id)
                                
                                // Create String
                                let dateString = "10/20/2019"
                                let startDateString = "01/01/1900"
                                
                                // Create Date Formatter
                                let dateFormatter = DateFormatter()
                                
                                // Set Date Format
                                dateFormatter.dateFormat = "M/d/yyyy"
                                
                                let days = Calendar.current.dateComponents([.day], from: dateFormatter.date(from: startDateString)!, to: dateFormatter.date(from: dateString)!).day!
                                
                                let inTime =  Int(Double(dateFormatter.date(from: dateString)!.timeIntervalSince1970))
                                
                                
                                
                                if (id == ""){
                                    print("There is no value")
                                }else{
                                    print("There is a value \(String(describing: id))")
                                    //let usedRangeURI = "https://graph.microsoft.com/v1.0/me/drive/items/\(id)/workbook/worksheets('Sheet1')/usedRange"
                                    //let usedRangeURI = "https://graph.microsoft.com/v1.0/me/drive/root:/elevator_experiments.xlsx:/workbook/worksheets/" + String(describing: id)  + "/range(address='Data Collection!A1:A20')"
                                    //let usedRangeURI = "https://graph.microsoft.com/v1.0/me/drive/root:/elevator_experiments.xlsx:/workbook/worksheets:('" + String(describing: id)  + "')/usedRange"
                                    let usedRangeURI = "https://graph.microsoft.com/v1.0/me/drive/root:/elevator_experiments.xlsx:/workbook/worksheets(%27%7B" + String(describing: id) + "%7D%27)/tables/Table1/rows"

                                    var usedRangeRequest = URLRequest(url: URL(string: usedRangeURI)!)
                                    
                                    usedRangeRequest.setValue("Bearer \(result.accessToken)", forHTTPHeaderField: "Authorization")
                                    self.accessToken = result.accessToken
                                    
                                    
                                    let task2 = URLSession.shared.dataTask(with: usedRangeRequest) { data, response, error in
                                        if let error = error {
                                            print("Couldn't get graph result: \(error)")
                                            return
                                        }
                                        if let httpResponse = response as? HTTPURLResponse {
                                            print(httpResponse.statusCode)
                                        }
                                        
                                        if let result = try? JSONSerialization.jsonObject(with: data!, options: []) as? [String: Any]
                                        {
                                            print(result)
                                            var json = result["value"] as? [[String: Any]]
                                            //Get last record
                                            let lastid = ((json?[json!.count - 1]["values"])  as! [[Any]])[0][0]
                                            let lastindex = json!.count - 1
                                            var postData: [String: Any] = ["values":[[""]]]
                                            var indexURI = usedRangeURI
                                            var httpMethod = "POST"
                                            //check if the last id = number of days
                                            if (lastid as! Int == days){
                                                //Then update the record
                                                print("update")
                                                var dataRecord = json?[json!.count - 1]["values"] as! [[Any]]
                                                dataRecord[0][3] = dataRecord[0][3] as! Int + 1
                                                print(dataRecord)
                                                //postData = ["index": lastindex, "values": dataRecord]
                                                
                                                postData = ["values": [[days ,dataRecord[0][1] ,dataRecord[0][2],dataRecord[0][3],nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil]]]
                                                
                                                //postData = ["values": dataRecord]
                                                print("Post Data= \(postData)")

                                                httpMethod = "PATCH"
                                                indexURI = indexURI + "/$/ItemAt%28index=\(lastindex)%29"
                                                
                                                //https://graph.microsoft.com/v1.0/me/drive/root:/elevator_experiments.xlsx:/workbook/worksheets(%27%7B283EDF75-D6B8-9F4F-9BD2-F29CC1EE9F97%7D%27)/tables/Table1/rows/$/ItemAt(index=36)

                                            } else {
                                                //The insert a new record
                                                print("insert")

                                                postData = ["values": [[days,0,0,1,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil]]]
                                                
                                                print("Post Data= \(postData)")
                                                httpMethod = "POST"
                                            }

                                            usedRangeRequest = URLRequest(url: URL(string: indexURI)!)
                                            
                                            usedRangeRequest.httpMethod = httpMethod
                                            usedRangeRequest.setValue("application/json; charset=utf-8", forHTTPHeaderField: "Content-Type")  // the request is JSON

                                            
                                            usedRangeRequest.setValue("Bearer \(self.accessToken)", forHTTPHeaderField: "Authorization")
                                            
                                            let jsonData = try? JSONSerialization.data(withJSONObject: postData)
                                            
                                            usedRangeRequest.httpBody = jsonData
                                            let task3 = URLSession.shared.dataTask(with: usedRangeRequest) { data, response, error in
                                                if let error = error {
                                                    print("Couldn't get graph result: \(error)")
                                                    return
                                                }
                                                if let resultx = try? JSONSerialization.jsonObject(with: data!, options: []) as? [String: Any]
                                                {
                                                        print("\(resultx)")
                                                }
                                                
                                                if let httpResponse1 = response as? HTTPURLResponse {
                                                    print(httpResponse1.statusCode)
                                                    print(data as Any)
                                                }
                                                //print("Added Result = \(result)")
                                                
                                            }
                                            
                                            task3.resume()

                                            print("X=\(lastid) \(lastindex)")
                                        } else {
                                            print("Error")
                                            return
                                        }
                                        
                                        
                                    }
                                    task2.resume()


                                }
                                return

                            }
                        }  else {
                            print("Couldn't deserialize result JSON")
                            self.accessToken = ""
                            return
                        }

                        
                    }
                    
                    task.resume()*/

                    
    //end
                    sender.setTitle("Sign Out", for: .normal)
                    sender.backgroundColor = .red

                    Task {
                        await helper.getLastExcelTableRow(excelFile: "elevator_experiments.xlsx", tableName: "Table1")
                        //print("Helper=\(helper.tableRow)")
                        //print(helper.tableRow[0])
                        helper.el1Prob = String(format: "%.2f", helper.tableRow[0][13] as! Double)
                        helper.el2Prob = String(format: "%.2f", helper.tableRow[0][14] as! Double)
                        helper.el3Prob = String(format: "%.2f", helper.tableRow[0][15] as! Double)
                        self.elProb[0]=helper.el1Prob
                        self.elProb[1]=helper.el2Prob
                        self.elProb[2]=helper.el3Prob
                    }
                }
            }
        /*}catch{
            self.accessToken = ""
            print(error.localizedDescription)
            return
        }*/
        

    }
}

struct MyCustomLogin: UIViewControllerRepresentable{
    
    @Binding var elProb: [String]

    init(value: Binding<[String]>){
        self._elProb = value
    }
    
    func updateUIViewController(_ uiViewController: MicrosoftViewController, context: Context) {
    }
    
    typealias UIViewControllerType = MicrosoftViewController
    
    func makeUIViewController(context: UIViewControllerRepresentableContext<MyCustomLogin>) -> MicrosoftViewController {
            return MicrosoftViewController(value: self._elProb)
    }
}
